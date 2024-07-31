#ifndef CUSTOM_ANT_ENV_H
#define CUSTOM_ANT_ENV_H

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

class CustomAntEnv {
public:
    CustomAntEnv(const char* model_path);
    ~CustomAntEnv();

    mjModel* m;         // MuJoCo model
    mjData* d;          // MuJoCo data
    mjvCamera cam;      // abstract camera
    mjvOption opt;      // visualization options
    mjvScene scn;       // abstract scene
    mjrContext con;     // custom GPU context

    bool button_left;
    bool button_middle;
    bool button_right;
    double lastx;
    double lasty;
};

// Function declarations
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);

#endif // CUSTOM_ANT_ENV_H
