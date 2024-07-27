#ifndef GLFW_WINDOW_MANAGER_H
#define GLFW_WINDOW_MANAGER_H

#include <GLFW/glfw3.h>
#include <mujoco.h>  // Directly include mujoco.h
#include <iostream>

class GLFWWindowManager {
public:
    GLFWWindowManager();
    ~GLFWWindowManager();

    bool initialize();
    void createWindow(int width, int height, const char* title);
    void render(mjModel* m, mjData* d);
    bool windowShouldClose();
    void pollEvents();

private:
    GLFWwindow* window;
    mjvScene scn;
    mjrContext con;

    static void glfwErrorCallback(int error, const char* description);
};

#endif // GLFW_WINDOW_MANAGER_H
