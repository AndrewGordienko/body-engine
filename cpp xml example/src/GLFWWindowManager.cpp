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
    mjv_freeScene(&scn);
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
    std::cout << "GLFW window created successfully" << std::endl;

    // Ensure the scene and context are correctly initialized
    mjv_makeScene(nullptr, &scn, 2000);  // Allocate 2000 elements for the scene
    mjr_makeContext(nullptr, &con, mjFONTSCALE_150);  // Ensure context is created
    std::cout << "MuJoCo scene and context created successfully" << std::endl;
}

void GLFWWindowManager::render(mjModel* m, mjData* d) {
    if (!window || !m || !d) {
        std::cerr << "Invalid pointers in render function" << std::endl;
        return;  // Add checks for null pointers
    }

    std::cout << "Rendering..." << std::endl;
    std::cout << "Model: " << m << ", Data: " << d << std::endl;
    mjv_updateScene(m, d, nullptr, nullptr, nullptr, mjCAT_ALL, &scn);
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    std::cout << "Rendered frame" << std::endl;
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
