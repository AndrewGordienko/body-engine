#include "GLFWWindowManager.h"

GLFWWindowManager::GLFWWindowManager() : window(nullptr) {
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
}

GLFWWindowManager::~GLFWWindowManager() {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
    mjr_freeContext(&con);
}

bool GLFWWindowManager::initialize() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwSetErrorCallback(glfwErrorCallback);
    return true;
}

void GLFWWindowManager::createWindow(int width, int height, const char* title) {
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);
}

void GLFWWindowManager::render(mjModel* m, mjData* d) {
    if (!window) return;

    mjv_updateScene(m, d, nullptr, nullptr, nullptr, mjCAT_ALL, &scn);
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
}

bool GLFWWindowManager::windowShouldClose() {
    return glfwWindowShouldClose(window);
}

void GLFWWindowManager::pollEvents() {
    glfwPollEvents();
}

void GLFWWindowManager::glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW Error: " << description << std::endl;
}
