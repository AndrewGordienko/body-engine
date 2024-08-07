#ifndef CUSTOM_ANT_ENV_H
#define CUSTOM_ANT_ENV_H

#include <vector>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>

class CustomAntEnv {
public:
    CustomAntEnv(const char* model_path);
    ~CustomAntEnv();

    Eigen::MatrixXd getObservation() const;
    double getReward() const;
    void setAction(const std::vector<double>& action);
    void render();
    bool should_close() const;

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

    std::vector<Eigen::Vector3d> targetPositions; // Store target positions

    Eigen::Vector3d getCreaturePosition(int creatureIdx) const; // Helper function to get creature position
    void printMuJoCoData() const; // Helper function to print MuJoCo data arrays
};

// Function declarations
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);

#endif // CUSTOM_ANT_ENV_H
