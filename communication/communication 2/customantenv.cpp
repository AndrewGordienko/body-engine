#include "customantenv.h"
#include "tinyxml2.h"
#include <iostream>
#include <Eigen/Dense>
#include <pybind11/stl.h>
#include <sstream>
#include <vector>

using namespace tinyxml2; // This allows you to use tinyxml2 classes without the namespace prefix

CustomAntEnv::CustomAntEnv(const char* model_path) {
    // Initialize GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // Create a visible GLFW window
    window = glfwCreateWindow(1200, 900, "MuJoCo Environment", NULL, NULL);
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
    if (root == nullptr) return; // Handle error

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

double CustomAntEnv::getReward() const {
    // Implement your reward calculation logic here
    return 0.0; // Placeholder
}

Eigen::MatrixXd CustomAntEnv::getObservation() const {
    const int OBSERVATION_DIMS_PER_CREATURE = MAX_LEGS * MAX_PARTS_PER_LEG * DATA_POINTS_PER_SUBPART + DISTANCE_TO_TARGET_DIMS + 3; // Adjust these constants as per your model

    // Initialize the observation matrix for all creatures
    Eigen::MatrixXd observations = Eigen::MatrixXd::Zero(NUM_CREATURES, OBSERVATION_DIMS_PER_CREATURE);

    for (int creatureIdx = 0; creatureIdx < NUM_CREATURES; ++creatureIdx) {
        int observationIndex = 0; // Reset for each creature

        // Process leg and part data
        for (int legIdx = 0; legIdx < MAX_LEGS; ++legIdx) {
            for (int partIdx = 0; partIdx < MAX_PARTS_PER_LEG; ++partIdx) {
                // Placeholder logic for physics index calculation
                int physicsIdx = calculatePhysicsIndex(creatureIdx, legIdx, partIdx); // Implement this function based on your simulation setup

                // Example placeholders for retrieving sensor data
                double angle = d->qpos[physicsIdx];
                double velocity = d->qvel[physicsIdx];
                double acceleration = d->sensordata[physicsIdx]; // Assuming acceleration data is stored similarly

                // Populate observations with the retrieved data
                observations(creatureIdx, observationIndex++) = angle;
                observations(creatureIdx, observationIndex++) = velocity;
                observations(creatureIdx, observationIndex++) = acceleration;
            }
        }

        // Adding distance to target data
        Eigen::Vector2d distanceToTarget = calculateDistanceToTarget(creatureIdx); // Implement this function based on your simulation setup
        observations(creatureIdx, observationIndex++) = distanceToTarget.x();
        observations(creatureIdx, observationIndex++) = distanceToTarget.y();

        // Gyroscope data integration
        int gyroIndex = creatureIdx * 3; // Gyroscope data index calculation
        double gyro_x = d->sensordata[gyroIndex];     // Gyro X
        double gyro_y = d->sensordata[gyroIndex + 1]; // Gyro Y
        double gyro_z = d->sensordata[gyroIndex + 2]; // Gyro Z

        observations(creatureIdx, observationIndex++) = gyro_x;
        observations(creatureIdx, observationIndex++) = gyro_y;
        observations(creatureIdx, observationIndex++) = gyro_z;
    }
    return observations;
}

Eigen::Vector3d CustomAntEnv::getCreaturePosition(int creatureIdx) const {
    // Assuming the root joint of each creature is the first element of its qpos
    int rootIndex = creatureIdx * MAX_LEGS * MAX_PARTS_PER_LEG * 3; // Adjust as necessary
    Eigen::Vector3d position(d->qpos[rootIndex], d->qpos[rootIndex + 1], d->qpos[rootIndex + 2]);
    return position;
}

int CustomAntEnv::calculatePhysicsIndex(int creatureIdx, int legIdx, int partIdx) const {
    return creatureIdx * MAX_LEGS * MAX_PARTS_PER_LEG + legIdx * MAX_PARTS_PER_LEG + partIdx;
}

Eigen::Vector2d CustomAntEnv::calculateDistanceToTarget(int creatureIdx) const {
    Eigen::Vector3d creaturePos = getCreaturePosition(creatureIdx);
    Eigen::Vector3d targetPos = flag_positions.row(creatureIdx);

    Eigen::Vector3d distanceVec = targetPos - creaturePos;
    return Eigen::Vector2d(distanceVec.x(), distanceVec.y());
}

void CustomAntEnv::setAction(const std::vector<double>& action) {
    // Placeholder for setting actions, should be connected to actuators
}

void CustomAntEnv::render() {
    // Step the simulation
    mj_step(m, d);

    // Update scene and render
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool CustomAntEnv::should_close() const {
    return glfwWindowShouldClose(window);
}

// Existing callback functions...

void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
        mj_resetData(env->m, env->d);
        mj_forward(env->m, env->d);
    }
}

void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));

    // update button state
    env->button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    env->button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    env->button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &env->lastx, &env->lasty);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));

    // no buttons down: nothing to do
    if (!env->button_left && !env->button_middle && !env->button_right) {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - env->lastx;
    double dy = ypos - env->lasty;
    env->lastx = xpos;
    env->lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (env->button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (env->button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(env->m, action, dx / height, dy / height, &env->scn, &env->cam);
}

void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
    mjv_moveCamera(env->m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &env->scn, &env->cam);
}
