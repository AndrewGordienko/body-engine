#include <cstdio>
#include <cstring>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include "GLFWWindowManager.h"
#include "customantenv.h"

// main function
int main(int argc, const char** argv) {
    // check command-line arguments
    if (argc != 2) {
        std::printf("USAGE: mujoco_renderer modelfile\n");
        return 0;
    }

    // create GLFW window manager
    GLFWWindowManager windowManager;
    GLFWwindow* window = windowManager.getWindow();

    // create environment
    CustomAntEnv env(argv[1]);

    // set user pointer for GLFW callbacks
    glfwSetWindowUserPointer(window, &env);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window)) {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = env.d->time;
        while (env.d->time - simstart < 1.0 / 60.0) {
            mj_step(env.m, env.d);
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(env.m, env.d, &env.opt, NULL, &env.cam, mjCAT_ALL, &env.scn);
        mjr_render(viewport, &env.scn, &env.con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    return 0;
}
