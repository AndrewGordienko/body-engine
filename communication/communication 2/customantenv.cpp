#include "customantenv.h"
#include "tinyxml2.h"
#include <iostream>
#include <Eigen/Dense>
#include <pybind11/stl.h>
#include <sstream>
#include <vector>

using namespace tinyxml2;

CustomAntEnv::CustomAntEnv(const char* model_path, int max_steps) : step_count(0), max_steps(max_steps) {
    // Initialize GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // Create a visible GLFW window
    window = glfwCreateWindow(900, 600, "MuJoCo Environment", NULL, NULL);
    if (!window) {
        glfwTerminate();
        mju_error("Could not create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Load and compile model
    char error[1000] = "Could not load binary model";
    if (std::strlen(model_path) > 4 && !std::strcmp(model_path + std::strlen(model_path) - 4, ".mjb")) {
        m = mj_loadModel(model_path, 0);
    } else {
        m = mj_loadXML(model_path, 0, error, 1000);
    }
    if (!m) {
        glfwDestroyWindow(window);
        glfwTerminate();
        mju_error("Load model error: %s", error);
    }

    // Make data
    d = mj_makeData(m);

    // Initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // Create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Set the GLFW window user pointer to this instance for callback access
    glfwSetWindowUserPointer(window, this);

    // Install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    button_left = false;
    button_middle = false;
    button_right = false;
    lastx = 0;
    lasty = 0;

    // Initialize flag positions from XML file
    initializeFlagPositionsFromXML("/Users/andrewgordienko/Documents/body engine/communication/communication 2/ant_model.xml");

    // Calculate intermediate targets after initializing flag positions
    calculateIntermediateTargets();
}
void CustomAntEnv::initializeFlagPositionsFromXML(const char* xml_file) {
    XMLDocument xmlDoc;
    XMLError eResult = xmlDoc.LoadFile(xml_file);
    if (eResult != XML_SUCCESS) {
        std::cerr << "Failed to load XML file. Error: " << xmlDoc.ErrorName() << std::endl;
        return;
    }

    // Initialize flag_positions matrix size
    flag_positions = Eigen::MatrixXd(NUM_CREATURES, 3);

    // Find all flag elements
    XMLNode* root = xmlDoc.FirstChild();
    if (root == nullptr) return;

    int flagIndex = 0;
    for (XMLElement* elem = root->FirstChildElement("geom"); elem != nullptr; elem = elem->NextSiblingElement("geom")) {
        const char* name = elem->Attribute("name");
        if (name != nullptr && std::string(name).find("flag_") == 0) {
            // Parse position attribute
            const char* posAttr = elem->Attribute("pos");
            std::vector<double> posValues = parsePosition(posAttr);

            // Assign to flag_positions matrix
            if (posValues.size() == 3 && flagIndex < NUM_CREATURES) {
                flag_positions.row(flagIndex) = Eigen::Vector3d(posValues[0], posValues[1], posValues[2]);
                ++flagIndex;
            }
        }
    }

    // Calculate intermediate targets after flag positions are initialized
    calculateIntermediateTargets();
}
void CustomAntEnv::loadNewModel(const std::string& xml_file) {
    if (d) {
        mj_deleteData(d);
        d = nullptr;
    }
    if (m) {
        mj_deleteModel(m);
        m = nullptr;
    }

    char error[1000] = "";
    m = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
    if (!m) {
        std::cerr << "Load model error: " << error << std::endl;
        throw std::runtime_error("Load model error.");
    }

    d = mj_makeData(m);
    reset(); // Reset the environment with the new model
}

void CustomAntEnv::reset() {
    step_count = 0;  // Reset the step count
    mj_resetData(m, d);  // Reset the MuJoCo data
    mj_forward(m, d);  // Forward the simulation to recompute state
}

std::vector<double> CustomAntEnv::parsePosition(const char* posAttr) {
    std::vector<double> pos;
    if (posAttr != nullptr) {
        std::stringstream ss(posAttr);
        double val;
        while (ss >> val) {
            pos.push_back(val);
            if (ss.peek() == ' ') ss.ignore();
        }
    }
    return pos;
}

CustomAntEnv::~CustomAntEnv() {
    // Free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    // Destroy the window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
}

Eigen::VectorXd CustomAntEnv::calculateReward() {
    Eigen::VectorXd rewards = Eigen::VectorXd::Zero(NUM_CREATURES);

    for (int creatureIdx = 0; creatureIdx < NUM_CREATURES; ++creatureIdx) {
        // Obtain the torso position and the flag position for the current creature
        Eigen::Vector3d torso_position = getCreaturePosition(creatureIdx);
        Eigen::Vector3d flag_pos = flag_positions.row(creatureIdx);
        double distanceToFinalFlag = (torso_position - flag_pos).norm();

        // Calculate the speed reward
        double speed_reward_factor = 1.0;
        double speed_reward = speed_reward_factor / (1 + d->time);

        // Calculate the energy penalty
        double energy_used = 0.0;
        for (int i = 0; i < m->nu; ++i) {
            energy_used += std::abs(d->ctrl[creatureIdx * m->nu + i]);
        }
        double energy_penalty = energy_used * 0.0005;

        // Calculate the flag reached reward
        double flag_reached_reward = (distanceToFinalFlag < 0.1) ? 10.0 : 0.0;

        // Calculate the intermediate reward
        auto [closestIndex, distanceToClosestIntermediate] = getClosestTargetIndexAndDistance(creatureIdx, torso_position);
        double intermediate_reward = 0.0;
        if (closestIndex != -1 && distanceToClosestIntermediate < 0.5) {
            intermediate_reward = (closestIndex + 1) * 10.0;
            intermediate_targets[creatureIdx].erase(intermediate_targets[creatureIdx].begin() + closestIndex);
        }

        // Calculate gyroscope-based stability reward
        int gyroIndex = creatureIdx * 3;
        double gyro_x = d->sensordata[gyroIndex];
        double gyro_y = d->sensordata[gyroIndex + 1];
        double gyro_z = d->sensordata[gyroIndex + 2];
        double gyro_stability_reward = -0.1 * (std::abs(gyro_x) + std::abs(gyro_y) + std::abs(gyro_z));

        // Combine all reward components
        rewards(creatureIdx) = speed_reward + flag_reached_reward - energy_penalty + intermediate_reward + gyro_stability_reward;
    }

    return rewards;
}

Eigen::MatrixXd CustomAntEnv::getObservation() const {
    const int OBSERVATION_DIMS_PER_CREATURE = MAX_LEGS * MAX_PARTS_PER_LEG * DATA_POINTS_PER_SUBPART + DISTANCE_TO_TARGET_DIMS + 3;

    // Initialize the observation matrix for all creatures
    Eigen::MatrixXd observations = Eigen::MatrixXd::Zero(NUM_CREATURES, OBSERVATION_DIMS_PER_CREATURE);

    for (int creatureIdx = 0; creatureIdx < NUM_CREATURES; ++creatureIdx) {
        int observationIndex = 0; // Reset for each creature

        // Process leg and part data
        for (int legIdx = 0; legIdx < MAX_LEGS; ++legIdx) {
            for (int partIdx = 0; partIdx < MAX_PARTS_PER_LEG; ++partIdx) {
                int physicsIdx = calculatePhysicsIndex(creatureIdx, legIdx, partIdx);

                double angle = d->qpos[physicsIdx];
                double velocity = d->qvel[physicsIdx];
                double acceleration = d->sensordata[physicsIdx];

                observations(creatureIdx, observationIndex++) = angle;
                observations(creatureIdx, observationIndex++) = velocity;
                observations(creatureIdx, observationIndex++) = acceleration;
            }
        }

        // Adding distance to target data
        Eigen::Vector2d distanceToTarget = calculateDistanceToTarget(creatureIdx);
        observations(creatureIdx, observationIndex++) = distanceToTarget.x();
        observations(creatureIdx, observationIndex++) = distanceToTarget.y();

        // Gyroscope data integration
        int gyroIndex = creatureIdx * 3;
        double gyro_x = d->sensordata[gyroIndex];
        double gyro_y = d->sensordata[gyroIndex + 1];
        double gyro_z = d->sensordata[gyroIndex + 2];

        observations(creatureIdx, observationIndex++) = gyro_x;
        observations(creatureIdx, observationIndex++) = gyro_y;
        observations(creatureIdx, observationIndex++) = gyro_z;
    }
    return observations;
}

Eigen::Vector3d CustomAntEnv::getCreaturePosition(int creatureIdx) const {
    int rootIndex = creatureIdx * MAX_LEGS * MAX_PARTS_PER_LEG * 3;
    Eigen::Vector3d position(d->qpos[rootIndex], d->qpos[rootIndex + 1], d->qpos[rootIndex + 2]);
    return position;
}

int CustomAntEnv::calculatePhysicsIndex(int creatureIdx, int legIdx, int partIdx) const {
    return creatureIdx * MAX_LEGS * MAX_PARTS_PER_LEG + legIdx * MAX_PARTS_PER_LEG + partIdx;
}

int CustomAntEnv::calculateControlIndex(int creatureIdx, int legIdx, int partIdx) const {
    int index = 0; // Initialize with the starting index for this creature

    // First, add offset for all previous creatures
    for (int prevCreature = 0; prevCreature < creatureIdx; ++prevCreature) {
        // Assuming each creature has the same number of motors, which is MAX_LEGS * MAX_PARTS_PER_LEG
        index += MAX_LEGS * MAX_PARTS_PER_LEG * CONTROLS_PER_PART;
    }

    // Then, calculate index for current creature
    for (int i = 0; i < legIdx; ++i) {
        index += MAX_PARTS_PER_LEG * CONTROLS_PER_PART;
    }
    index += partIdx * CONTROLS_PER_PART;

    return index;
}

Eigen::Vector2d CustomAntEnv::calculateDistanceToTarget(int creatureIdx) const {
    Eigen::Vector3d creaturePos = getCreaturePosition(creatureIdx);
    Eigen::Vector3d targetPos = flag_positions.row(creatureIdx);

    Eigen::Vector3d distanceVec = targetPos - creaturePos;
    return Eigen::Vector2d(distanceVec.x(), distanceVec.y());
}

void CustomAntEnv::setAction(const Eigen::MatrixXd& actions) {
    // Assuming actions is a NUM_CREATURES x (MAX_LEGS * MAX_PARTS_PER_LEG) Eigen::MatrixXd
    for (int creatureIdx = 0; creatureIdx < NUM_CREATURES; ++creatureIdx) {
        for (int legIdx = 0; legIdx < MAX_LEGS; ++legIdx) {
            for (int partIdx = 0; partIdx < MAX_PARTS_PER_LEG; ++partIdx) {
                int actionIndex = legIdx * MAX_PARTS_PER_LEG + partIdx;
                if (actionIndex < actions.cols()) {
                    int controlIndex = calculateControlIndex(creatureIdx, legIdx, partIdx);
                    if (controlIndex >= 0) { // Check if the motor exists
                        double actionValue = actions(creatureIdx, actionIndex);
                        d->ctrl[controlIndex] = actionValue;
                    }
                }
            }
        }
    }
}

void CustomAntEnv::render() {
    mj_step(m, d);

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();
    step_count++;  // Increment step count on each render
}

bool CustomAntEnv::should_close() const {
    return glfwWindowShouldClose(window);
}

bool CustomAntEnv::isDone() const {
    return step_count >= max_steps;
}

void CustomAntEnv::calculateIntermediateTargets() {
    intermediate_targets.clear();

    if (flag_positions.size() <= 0) {
        std::cerr << "Flag positions not initialized.\n";
        return;
    }

    for (int i = 0; i < NUM_CREATURES; ++i) {
        if (i >= flag_positions.rows()) {
            std::cerr << "Index out of bounds for flag_positions matrix.\n";
            continue;
        }

        Eigen::Vector3d spawn_pos = getCreaturePosition(i);
        Eigen::Vector3d target_pos = flag_positions.row(i);

        std::vector<Eigen::Vector3d> creatureTargets;
        for (int j = 1; j <= 5; ++j) {
            double fraction = static_cast<double>(j) / 6.0;
            Eigen::Vector3d intermediateTarget = spawn_pos + fraction * (target_pos - spawn_pos);
            creatureTargets.push_back(intermediateTarget);
        }
        intermediate_targets.push_back(creatureTargets);
    }
}

std::pair<int, double> CustomAntEnv::getClosestTargetIndexAndDistance(int creature_id, const Eigen::Vector3d& position) const {
    if (creature_id >= intermediate_targets.size()) {
        std::cerr << "Error: creature_id out of bounds.\n";
        return {-1, std::numeric_limits<double>::max()};
    }

    int closestIndex = -1;
    double closestDistance = std::numeric_limits<double>::max();

    for (size_t i = 0; i < intermediate_targets[creature_id].size(); ++i) {
        double distance = (intermediate_targets[creature_id][i] - position).norm();
        if (distance < closestDistance) {
            closestDistance = distance;
            closestIndex = static_cast<int>(i);
        }
    }
    return {closestIndex, closestDistance};
}


// Existing callback functions...

void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
        mj_resetData(env->m, env->d);
        mj_forward(env->m, env->d);
    }
}

void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));

    env->button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    env->button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    env->button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    glfwGetCursorPos(window, &env->lastx, &env->lasty);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));

    if (!env->button_left && !env->button_middle && !env->button_right) {
        return;
    }

    double dx = xpos - env->lastx;
    double dy = ypos - env->lasty;
    env->lastx = xpos;
    env->lasty = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (env->button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (env->button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(env->m, action, dx / height, dy / height, &env->scn, &env->cam);
}

void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
    mjv_moveCamera(env->m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &env->scn, &env->cam);
}
