#ifndef CUSTOM_ANT_ENV_H
#define CUSTOM_ANT_ENV_H

#include <vector>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include "tinyxml2.h" // Include tinyxml2 for XML parsing

class CustomAntEnv {
public:
    CustomAntEnv(const char* model_path, int max_steps);
    ~CustomAntEnv();

    Eigen::MatrixXd getObservation() const;
    Eigen::VectorXd calculateReward(); // Removed const
    void setAction(const Eigen::MatrixXd& actions); // Updated parameter type
    void render();
    bool should_close() const;
    bool isDone() const;  // Add isDone method

    int calculatePhysicsIndex(int creatureIdx, int legIdx, int partIdx) const;
    Eigen::Vector2d calculateDistanceToTarget(int creatureIdx) const;

    mjModel* m;         // MuJoCo model
    mjData* d;          // MuJoCo data
    mjvCamera cam;      // abstract camera
    mjvOption opt;      // visualization options
    mjvScene scn;       // abstract scene
    mjrContext con;     // custom GPU context
    GLFWwindow* window; // GLFW window

    bool button_left;
    bool button_middle;
    bool button_right;
    double lastx;
    double lasty;

private:
    static constexpr int MAX_LEGS = 4;
    static constexpr int MAX_PARTS_PER_LEG = 3;
    static constexpr int DATA_POINTS_PER_SUBPART = 3;
    static constexpr int NUM_CREATURES = 9;
    static constexpr int DISTANCE_TO_TARGET_DIMS = 2; // Define this constant
    static constexpr int CONTROLS_PER_PART = 1; // Assuming one control per part

    Eigen::MatrixXd flag_positions; // Store flag positions
    std::vector<std::vector<Eigen::Vector3d>> intermediate_targets; // Store intermediate targets for each creature

    Eigen::Vector3d getCreaturePosition(int creatureIdx) const; // Helper function to get creature position
    void initializeFlagPositionsFromXML(const char* xml_file); // Initialize flag positions from XML
    std::vector<double> parsePosition(const char* posAttr); // Helper function to parse position attribute

    std::pair<int, double> getClosestTargetIndexAndDistance(int creatureIdx, const Eigen::Vector3d& torso_position) const;

    int step_count;     // Add step count
    int max_steps;      // Add max steps

    int calculateControlIndex(int creatureIdx, int legIdx, int partIdx) const;
};

// Function declarations
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);

#endif // CUSTOM_ANT_ENV_H
