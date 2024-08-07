#ifndef GLFW_WINDOW_MANAGER_H
#define GLFW_WINDOW_MANAGER_H

#include <GLFW/glfw3.h>
#include <stdexcept>

class GLFWWindowManager {
public:
    GLFWWindowManager();
    ~GLFWWindowManager();

    GLFWwindow* getWindow() { return window; }

    // Prevent copying and assignment
    GLFWWindowManager(const GLFWWindowManager&) = delete;
    GLFWWindowManager& operator=(const GLFWWindowManager&) = delete;

private:
    GLFWwindow* window;

    void initializeGLFW();
    void createWindow();
    void destroyWindow();
};

#endif // GLFW_WINDOW_MANAGER_H
