#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>
#include <set>
#include <execution>
#include <algorithm>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

const int CANVAS_RES = 200;
const float FLOOR_SIZE = 20.0f;
bool canvas[CANVAS_RES][CANVAS_RES] = { false };

float brushColor[3] = { 0.0f, 0.4f, 0.9f }; // Default Blue
float brushSize = 1.0f;
float fluidViscosity = 0.2f;
float fluidDiffusion = 0.0001f;

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 force;
    float mass = 4.0f;
};

struct Spring {
    int p1, p2;     // Indices of connected particles
    float restLen;
    float stiffness = 20000.0f;
    float damping = 200.0f;
};

struct FluidGrid {
    glm::vec3 density[CANVAS_RES][CANVAS_RES];
    glm::vec2 velocity[CANVAS_RES][CANVAS_RES];
    // For Eulerian, we usually need "previous" states for stable calculation
    glm::vec3 density_prev[CANVAS_RES][CANVAS_RES];
    glm::vec2 velocity_prev[CANVAS_RES][CANVAS_RES];
};
FluidGrid paintFluid;

class FluidSolver {
public:
    FluidSolver() {
        // Initialize the index helper once
        for (int i = 0; i < CANVAS_RES; i++) {
            indices[i] = i;
        }
    }

    void step(FluidGrid& grid, float dt) {
        float visco = fluidViscosity; // Viscosity: how thick the liquid is
        float diff = fluidDiffusion;  // Diffusion: how fast the paint spreads

        // 1. Velocity Step
        // Add forces (if any), diffuse velocity, then ensure incompressibility
        diffuse2D(grid.velocity_prev, grid.velocity, visco, dt);
        project(grid.velocity_prev, grid.velocity); // Project stores result in velocity_prev

        // Move velocity along itself
        advect2D(grid.velocity, grid.velocity_prev, grid.velocity_prev, dt);
        project(grid.velocity, grid.velocity_prev);

        // 2. Density Step
        // Diffuse the paint density, then move it along the velocity field
        diffuseDensity(grid.density_prev, grid.density, diff, dt);
        advectDensity(grid.density, grid.density_prev, grid.velocity, dt);
    }

private:
    int indices[CANVAS_RES];
    // Solve linear equations (Gauss-Seidel) for diffusion and projection
    void lin_solve_vec3(glm::vec3 x[CANVAS_RES][CANVAS_RES], glm::vec3 x0[CANVAS_RES][CANVAS_RES], float a, float c) {
        for (int k = 0; k < 20; k++) {
            std::for_each(std::execution::par_unseq, &indices[1], &indices[CANVAS_RES - 1], [&](int i) {
                for (int j = 1; j < CANVAS_RES - 1; j++) {
                    // Vector math handles R, G, and B simultaneously
                    x[i][j] = (x0[i][j] + a * (x[i - 1][j] + x[i + 1][j] + x[i][j - 1] + x[i][j + 1])) / c;
                }
                });
        }
    }

    void diffuseDensity(glm::vec3 x[CANVAS_RES][CANVAS_RES], glm::vec3 x0[CANVAS_RES][CANVAS_RES], float diff, float dt) {
        float a = dt * diff * (CANVAS_RES - 2) * (CANVAS_RES - 2);
        // Overload lin_solve or manually solve per channel
        lin_solve_vec3(x, x0, a, 1 + 4 * a);
    }

    void diffuse2D(glm::vec2 x[CANVAS_RES][CANVAS_RES], glm::vec2 x0[CANVAS_RES][CANVAS_RES], float diff, float dt) {
        float a = dt * diff * (CANVAS_RES - 2) * (CANVAS_RES - 2);
        // Solve X and Y components of velocity independently
        for (int k = 0; k < 20; k++) {
            for (int i = 1; i < CANVAS_RES - 1; i++) {
                for (int j = 1; j < CANVAS_RES - 1; j++) {
                    x[i][j].x = (x0[i][j].x + a * (x[i - 1][j].x + x[i + 1][j].x + x[i][j - 1].x + x[i][j + 1].x)) / (1 + 4 * a);
                    x[i][j].y = (x0[i][j].y + a * (x[i - 1][j].y + x[i + 1][j].y + x[i][j - 1].y + x[i][j + 1].y)) / (1 + 4 * a);
                }
            }
        }
    }

    void advectDensity(glm::vec3 d[CANVAS_RES][CANVAS_RES], glm::vec3 d0[CANVAS_RES][CANVAS_RES], glm::vec2 v[CANVAS_RES][CANVAS_RES], float dt) {
        float dt0 = dt * (CANVAS_RES - 2);
        for (int i = 1; i < CANVAS_RES - 1; i++) {
            for (int j = 1; j < CANVAS_RES - 1; j++) {
                float x = (float)i - dt0 * v[i][j].x;
                float y = (float)j - dt0 * v[i][j].y;

                if (x < 0.5f) x = 0.5f; if (x > CANVAS_RES - 1.5f) x = CANVAS_RES - 1.5f;
                int i0 = (int)x; int i1 = i0 + 1;
                if (y < 0.5f) y = 0.5f; if (y > CANVAS_RES - 1.5f) y = CANVAS_RES - 1.5f;
                int j0 = (int)y; int j1 = j0 + 1;

                float s1 = x - i0; float s0 = 1.0f - s1;
                float t1 = y - j0; float t0 = 1.0f - t1;

                // Linearly interpolate the RGB color vectors
                d[i][j] = s0 * (t0 * d0[i0][j0] + t1 * d0[i0][j1]) +
                    s1 * (t0 * d0[i1][j0] + t1 * d0[i1][j1]);
            }
        }
    }

    // Similar to advectDensity but for the velocity vector itself
    void advect2D(glm::vec2 v[CANVAS_RES][CANVAS_RES], glm::vec2 v0[CANVAS_RES][CANVAS_RES], glm::vec2 vel_field[CANVAS_RES][CANVAS_RES], float dt) {
        float dt0 = dt * (CANVAS_RES - 2);
        for (int i = 1; i < CANVAS_RES - 1; i++) {
            for (int j = 1; j < CANVAS_RES - 1; j++) {
                float x = (float)i - dt0 * vel_field[i][j].x;
                float y = (float)j - dt0 * vel_field[i][j].y;

                if (x < 0.5f) x = 0.5f; if (x > CANVAS_RES - 1.5f) x = CANVAS_RES - 1.5f;
                int i0 = (int)x; int i1 = i0 + 1;
                if (y < 0.5f) y = 0.5f; if (y > CANVAS_RES - 1.5f) y = CANVAS_RES - 1.5f;
                int j0 = (int)y; int j1 = j0 + 1;

                float s1 = x - i0; float s0 = 1 - s1;
                float t1 = y - j0; float t0 = 1 - t1;

                v[i][j] = s0 * (t0 * v0[i0][j0] + t1 * v0[i0][j1]) +
                    s1 * (t0 * v0[i1][j0] + t1 * v0[i1][j1]);
            }
        }
    }

    void project(glm::vec2 v[CANVAS_RES][CANVAS_RES], glm::vec2 p_div[CANVAS_RES][CANVAS_RES]) {
        float h = 1.0f / CANVAS_RES; // Grid spacing

        for (int i = 1; i < CANVAS_RES - 1; i++) {
            for (int j = 1; j < CANVAS_RES - 1; j++) {
                // Correct Divergence formula (notice we use h, not RES)
                p_div[i][j].x = -0.5f * h * (v[i + 1][j].x - v[i - 1][j].x + v[i][j + 1].y - v[i][j - 1].y);
                p_div[i][j].y = 0; // Initial guess for pressure
            }
        }

        // Solve for pressure
        for (int k = 0; k < 20; k++) {
            for (int i = 1; i < CANVAS_RES - 1; i++) {
                for (int j = 1; j < CANVAS_RES - 1; j++) {
                    p_div[i][j].y = (p_div[i][j].x + p_div[i - 1][j].y + p_div[i + 1][j].y +
                        p_div[i][j - 1].y + p_div[i][j + 1].y) / 4.0f;
                }
            }
        }

        // Subtract pressure gradient
        for (int i = 1; i < CANVAS_RES - 1; i++) {
            for (int j = 1; j < CANVAS_RES - 1; j++) {
                // Scale velocity update by 1/h
                v[i][j].x -= 0.5f * (p_div[i + 1][j].y - p_div[i - 1][j].y) / h;
                v[i][j].y -= 0.5f * (p_div[i][j + 1].y - p_div[i][j - 1].y) / h;
            }
        }
    }
};



// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 20.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -90.0f);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Buffer for rendering the points
unsigned int paintVAO, paintVBO;
std::vector<glm::vec3> paintPositions;

std::vector<Particle> particles;
std::vector<Spring> springs;

void addDensity(int x, int y, glm::vec3 color, float amount) {
    // Only update the specific cells under the brush
    paintFluid.density[x][y] += color * amount;
}
glm::vec3 getMouseWorldPos(GLFWwindow* window, glm::mat4 projection, glm::mat4 view) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Normalize coordinates to [-1, 1]
    float x = (2.0f * (float)xpos) / SCR_WIDTH - 1.0f;
    float y = 1.0f - (2.0f * (float)ypos) / SCR_HEIGHT;

    // In Ortho, the ray direction is simply the camera's Front vector
    glm::vec3 ray_dir = camera.Front;

    // The ray origin starts on the "near plane" at the mouse coordinate
    glm::mat4 invProjView = glm::inverse(projection * view);
    glm::vec4 screenPos = glm::vec4(x, y, -1.0f, 1.0f);
    glm::vec4 worldPos = invProjView * screenPos;
    glm::vec3 ray_origin = glm::vec3(worldPos) / worldPos.w;

    // Ray-Plane Intersection (Y = 0)
    // t = -origin.y / direction.y
    float t = -ray_origin.y / ray_dir.y;
    return ray_origin + t * ray_dir;
}

void initTetrahedralBrush(float scale, glm::vec3 offset) {
    particles.clear();
    springs.clear();

    // Raw mesh data from PaintBrush.obj
    std::vector<glm::vec3> meshPositions = {
        {-0.058779, -0.089477, -4.023622}, { -0.058779, 7.643711, -2.476540},
        { 3.425780, -0.089477, -2.011811}, {  2.085968, 7.643711, -1.238270},
        { 3.425780, -0.089477,  2.011811}, {  2.085968, 7.643711,  1.238270},
        {-0.058779, -0.089477,  4.023622}, { -0.058779, 7.643711,  2.476540},
        {-3.543338, -0.089477,  2.011811}, { -2.203526, 7.643711,  1.238270},
        {-3.543338, -0.089477, -2.011811}, { -2.203526, 7.643711, -1.238270},
        {-0.058779,  4.248201, -3.395509}, {  2.881818, 4.248201, -1.697755},
        { 2.881818,  4.248201,  1.697755}, { -0.058779, 4.248201,  3.395509},
        {-2.999376,  4.248201,  1.697755}, { -2.999376, 4.248201, -1.697755}
    };

    for (const auto& pos : meshPositions) {

        Particle p;
        // Apply Scale first, then Translation (Position)[cite: 4]
        glm::vec3 pLocal = pos;

        float minY = -0.089477f;
        float maxY = 7.643711f;
        float t = (pLocal.y - minY) / (maxY - minY);

        // Taper factor (adjust these two values)
        float bottomScale = 0.3f;   // how narrow the tip is
        float topScale = 1.0f;   // keep top unchanged

        float radiusScale = glm::mix(bottomScale, topScale, t);

        // Apply scaling ONLY on XZ (radius), not Y
        pLocal.x *= radiusScale;
        pLocal.z *= radiusScale;

        p.position = (pLocal * scale) + offset;
        p.velocity = glm::vec3(0.0f);
        p.force = glm::vec3(0.0f);
        p.mass = 20.0f;
        particles.push_back(p);
    }

    std::set<std::pair<int, int>> springSet;

    auto addSpring = [&](int i, int j) {
        if (i > j) std::swap(i, j);
        if (springSet.count({ i,j })) return;

        springSet.insert({ i,j });
        float dist = glm::distance(particles[i].position, particles[j].position);
        springs.push_back({ i, j, dist, 1500.0f, 60.0f });
    };

    // 3. Connectivity from OBJ Faces
    // Note: OBJ indices start at 1, so we subtract 1
    std::vector<int> meshIndices = {
        6,4,18, 14,13,18, 13,2,18, 17,7,18, 16,7,18, 14,5,18, 4,10,6,
        17,10,18, 10,8,18, 10,6,18, 15,5,18, 11,1,14, 12,10,18, 9,17,11,
        7,18,11, 11,3,14, 15,14,18, 17,11,18, 14,4,18, 8,18,16, 15,18,16,
        17,8,18, 17,16,18, 1,18,14, 6,2,4, 14,4,13, 13,4,2, 17,16,7,
        16,5,7, 14,11,5, 4,2,10, 17,12,10, 10,6,8, 10,2,6, 15,14,5,
        11,3,1, 12,2,10, 9,7,17, 7,5,18, 11,5,3, 15,6,14, 17,7,11,
        14,6,4, 8,6,18, 16,6,15, 17,10,8, 17,8,16, 1,11,18, 13,1,18
    };

    for (size_t i = 0; i < meshIndices.size(); i += 3) {
        addSpring(meshIndices[i] - 1, meshIndices[i + 1] - 1);
        addSpring(meshIndices[i + 1] - 1, meshIndices[i + 2] - 1);
        addSpring(meshIndices[i + 2] - 1, meshIndices[i] - 1);
    }

    // 4. Internal "Volume" Spring - Set to TOP instead of average center
    float maxY = -999.0f;
    for (auto& p : particles) {
        if (p.position.y > maxY) maxY = p.position.y;
    }

    // Position the center particle at the very top of the brush mesh
    // We use offset.x and offset.z so it's centered horizontally
    glm::vec3 topCenter(offset.x, maxY, offset.z);

    int centerIdx = particles.size(); // This index will be the very last particle
    particles.push_back({ topCenter, glm::vec3(0), glm::vec3(0), 1.0f });

    // Connect every vertex to this top point to create the "handle" structure
    for (int i = 0; i < centerIdx; ++i) {
        // Increase stiffness significantly for the handle-to-tip connection
        float handleStiffness = 8000.0f;
        float handleDamping = 250.0f;
        //float dist = glm::distance(particles[i].position, particles[centerIdx].position);
        float dist = glm::distance(particles[i].position, particles[centerIdx].position);
        printf("restlen = %f\n", dist);
        springs.push_back({ i, centerIdx, dist, handleStiffness, handleDamping });
    }
}

void resetCanvas(FluidGrid& grid) {
    memset(grid.density, 0, sizeof(grid.density));
    memset(grid.density_prev, 0, sizeof(grid.density_prev));
    memset(grid.velocity, 0, sizeof(grid.velocity));
    memset(grid.velocity_prev, 0, sizeof(grid.velocity_prev));
}

void saveCanvas(FluidGrid& grid) {
    unsigned char* pixels = new unsigned char[CANVAS_RES * CANVAS_RES * 3];

    for (int y = 0; y < CANVAS_RES; y++) {
        for (int x = 0; x < CANVAS_RES; x++) {
            auto color = grid.density[x][y];

            // 180-degree rotation logic:
            // Invert the x and y indices used for the output buffer
            int invX = (CANVAS_RES - 1) - x;
            int invY = (CANVAS_RES - 1) - y;

            // Calculate the index using the inverted coordinates
            int idx = (invY * CANVAS_RES + invX) * 3;

            pixels[idx + 0] = (unsigned char)(std::min(color.r, 1.0f) * 255);
            pixels[idx + 1] = (unsigned char)(std::min(color.g, 1.0f) * 255);
            pixels[idx + 2] = (unsigned char)(std::min(color.b, 1.0f) * 255);
        }
    }

    stbi_write_png("canvas_output.png", CANVAS_RES, CANVAS_RES, 3, pixels, CANVAS_RES * 3);

    delete[] pixels;
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_CAPTURED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    stbi_set_flip_vertically_on_load(true);

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // build and compile shaders
    // -------------------------
    Shader ourShader("1.model_loading.vs", "1.model_loading.fs");
    Shader paintShader("paint.vs", "paint.fs");

    // 1. Initialize physics data
    initTetrahedralBrush(1.0f, glm::vec3(0.0f, 10.0f, 0.0f));

    // 2. Setup VAO/VBO for lines
    unsigned int lineVAO, lineVBO;
    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    // Reserve space for all spring vertices (2 vertices per spring, 3 floats per vertex)
    glBufferData(GL_ARRAY_BUFFER, springs.size() * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // 5. Draw a simple floor cross for reference
    std::vector<float> floorVertices = {
        // First Triangle
        -FLOOR_SIZE, 0.0f, -FLOOR_SIZE,
         FLOOR_SIZE, 0.0f, -FLOOR_SIZE,
         FLOOR_SIZE, 0.0f,  FLOOR_SIZE,
         // Second Triangle
         FLOOR_SIZE, 0.0f,  FLOOR_SIZE,
         -FLOOR_SIZE, 0.0f,  FLOOR_SIZE,
         -FLOOR_SIZE, 0.0f, -FLOOR_SIZE
    };
    unsigned int floorVAO, floorVBO;
    glGenVertexArrays(1, &floorVAO);
    glGenBuffers(1, &floorVBO);
    glBindVertexArray(floorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
    glBufferData(GL_ARRAY_BUFFER, floorVertices.size() * sizeof(float), floorVertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glGenVertexArrays(1, &paintVAO);
    glGenBuffers(1, &paintVBO);
    glBindVertexArray(paintVAO);
    glBindBuffer(GL_ARRAY_BUFFER, paintVBO);
    // We'll update this dynamically, so start with NULL or a small size
    glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // 1. Create the texture
    unsigned int fluidTexture;
    glGenTextures(1, &fluidTexture);
    glBindTexture(GL_TEXTURE_2D, fluidTexture);

    // Use GL_RED or GL_LUMINANCE since density is a single float
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, CANVAS_RES, CANVAS_RES, 0, GL_RGB, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 2. Setup a Quad (Floor) VAO that includes Texture Coordinates
    float quadVertices[] = {
        // positions            // texCoords
        -FLOOR_SIZE, 0.01f, -FLOOR_SIZE,  0.0f, 0.0f,
         FLOOR_SIZE, 0.01f, -FLOOR_SIZE,  1.0f, 0.0f,
         FLOOR_SIZE, 0.01f,  FLOOR_SIZE,  1.0f, 1.0f,
        -FLOOR_SIZE, 0.01f,  FLOOR_SIZE,  0.0f, 1.0f
    };
    unsigned int quadIndices[] = { 0, 1, 2, 0, 2, 3 };

    unsigned int quadVAO, quadVBO, quadEBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    FluidSolver fluidSolver;


    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- PHYSICS UPDATE ---
        float substeps = 10;
        float subDelta = (deltaTime > 0.1f ? 0.1f : deltaTime) / substeps; // Cap dt for stability

        float orthoSize = 15.0f; // Increase this to see more of the floor
        float aspect = (float)SCR_WIDTH / (float)SCR_HEIGHT;

        glm::mat4 projection = glm::ortho(
            -orthoSize * aspect,  // Left
            orthoSize * aspect,  // Right
            -orthoSize,           // Bottom
            orthoSize,           // Top
            0.1f,                // Near
            100.0f               // Far
        );
        glm::mat4 view = camera.GetViewMatrix();
        glm::vec3 mouseWorld = getMouseWorldPos(window, projection, view);

        float hoverHeight = 10.0f; // Height above plane
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            hoverHeight = 0.0f; // Press down
        }

        float constMass = 5.0f;
        float gravity = -1.0f;
        float forceY = gravity * constMass;
        // 3. Update the handle (the volume center particle)
        int handleIdx = particles.size() - 1;
        particles[handleIdx].position = mouseWorld + glm::vec3(0, hoverHeight, 0);
        particles[handleIdx].velocity = glm::vec3(0);

        // 1. Pre-calculate values outside the substeps loop
        float invFloorRange = 1.0f / (2.0f * FLOOR_SIZE);

        for (int s = 0; s < substeps; ++s) {
            // 2. Combined Force & Reset (Optional: use memset for forces if using SoA)
            for (int i = 0; i < particles.size(); ++i) {
                if (i == handleIdx) {
                    particles[i].force = glm::vec3(0); // Ensure handle doesn't accumulate force
                    continue;
                }
                particles[i].force = glm::vec3(0.0f, forceY * particles[i].mass, 0.0f);
                
            }

            // 3. Optimized Spring Loop
            for (auto& sp : springs) {
                glm::vec3 diff = particles[sp.p1].position - particles[sp.p2].position;
                float distSq = glm::dot(diff, diff);

                if (distSq > 0.0001f) {
                    float dist = sqrt(distSq);
                    glm::vec3 dir = diff / dist;

                    // Fused spring and damping calculation
                    float springF = -sp.stiffness * (dist - sp.restLen);
                    float dampF = glm::dot(particles[sp.p1].velocity - particles[sp.p2].velocity, dir) * sp.damping;

                    glm::vec3 totalF = (springF - dampF) * dir;
                    particles[sp.p1].force += totalF;
                    particles[sp.p2].force -= totalF;
                }
            }

            //printf("particle size = %d", particles.size());
            // 4. Integration & Floor logic
            for (int i = 0; i < particles.size(); ++i) {
                if (i == handleIdx) continue;

                particles[i].velocity += (particles[i].force / particles[i].mass) * subDelta;
                particles[i].velocity *= 0.95f;
                particles[i].position += particles[i].velocity * subDelta;

                //particles[i].velocity.y = std::clamp(particles[i].velocity.y, -.0f, 100.0f);
                /*if (glm::length(particles[i].velocity) > 20.0f)
                    printf("particles[i].velocity = %f %f %f\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);*/

                if (particles[i].position.y < 0.05f) {
                    particles[i].position.y = 0.05f;
                    particles[i].velocity.y = 0.0f;
                    particles[i].velocity.x *= 0.9f;
                    particles[i].velocity.z *= 0.9f;

                    // Calculate squared velocity for the "is moving" check
                    float velSq = glm::dot(particles[i].velocity, particles[i].velocity);

                    if (velSq > 0.01f) { // Threshold check using squared value (no sqrt needed)
                        int cx = (int)((particles[i].position.x + FLOOR_SIZE) * invFloorRange * CANVAS_RES);
                        int cz = (int)((particles[i].position.z + FLOOR_SIZE) * invFloorRange * CANVAS_RES);

                        if (cx >= 0 && cx < CANVAS_RES && cz >= 0 && cz < CANVAS_RES) {
                            // Multiply the 'amount' by the brushColor to store the actual RGB data
                            glm::vec3 colorToApply = glm::make_vec3(brushColor) * 0.1f * brushSize;
                            paintFluid.density[cx][cz] += colorToApply;

                            paintFluid.velocity[cx][cz] += glm::vec2(particles[i].velocity.x, particles[i].velocity.z);
                        }
                    }
                }
            }
        }
        // 5. Move fluid solver outside substeps for a huge speedup
        fluidSolver.step(paintFluid, deltaTime);

        // --- PREPARE DATA ---
        std::vector<float> lineVertices;
        for (const auto& sp : springs) {
            lineVertices.push_back(particles[sp.p1].position.x);
            lineVertices.push_back(particles[sp.p1].position.y);
            lineVertices.push_back(particles[sp.p1].position.z);
            lineVertices.push_back(particles[sp.p2].position.x);
            lineVertices.push_back(particles[sp.p2].position.y);
            lineVertices.push_back(particles[sp.p2].position.z);
        }

        // --- RENDER ---
        ourShader.use();

        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);
        ourShader.setMat4("model", glm::mat4(1.0f));

        // Draw Floor
        ourShader.setVec3("objectColor", glm::vec3(0.5f, 0.5f, 0.5f));
        glBindVertexArray(floorVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        //std::vector<glm::vec3> gridRenderPos;
        //for (int x = 0; x < CANVAS_RES; x++) {
        //    for (int z = 0; z < CANVAS_RES; z++) {
        //        if (paintFluid.density[x][z] > 0.01f) {
        //            // Map grid indices back to world space
        //            float worldX = ((float)x / CANVAS_RES) * (2.0f * FLOOR_SIZE) - FLOOR_SIZE;
        //            float worldZ = ((float)z / CANVAS_RES) * (2.0f * FLOOR_SIZE) - FLOOR_SIZE;
        //            gridRenderPos.push_back(glm::vec3(worldX, 0.01f, worldZ));
        //        }
        //    }
        //}

        //if (!gridRenderPos.empty()) {
        //    ourShader.setVec3("objectColor", glm::vec3(0.0f, 0.5f, 1.0f)); // Blue ink
        //    glBindVertexArray(paintVAO);
        //    glBindBuffer(GL_ARRAY_BUFFER, paintVBO);
        //    glBufferData(GL_ARRAY_BUFFER, gridRenderPos.size() * sizeof(glm::vec3), gridRenderPos.data(), GL_DYNAMIC_DRAW);
        //    glPointSize(4.0f);
        //    glDrawArrays(GL_POINTS, 0, (GLsizei)gridRenderPos.size());
        //}
        
       // 1. Upload the fresh density data from CPU to GPU
        glBindTexture(GL_TEXTURE_2D, fluidTexture);
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0, 0, 0,
            CANVAS_RES, CANVAS_RES,
            GL_RGB,        // Change from GL_RED
            GL_FLOAT,      // Data type
            paintFluid.density
        );

        // 2. Now draw with the shader
        paintShader.use();
        paintShader.setMat4("projection", projection);
        paintShader.setMat4("view", view);
        paintShader.setMat4("model", glm::mat4(1.0f));
        paintShader.setInt("fluidTexture", 0);
        paintShader.setVec3("inkColor", glm::make_vec3(brushColor));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fluidTexture);

        glBindVertexArray(quadVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        ourShader.use();
        // Draw Cylinder (Update buffer then draw)
        ourShader.setVec3("objectColor", glm::vec3(1.0f, 1.0f, 1.0f));
        glBindVertexArray(lineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, lineVertices.size() * sizeof(float), lineVertices.data());
        glDrawArrays(GL_LINES, 0, (GLsizei)(lineVertices.size() / 3));

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(550, 0), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_Once);

        ImGui::Begin("aint Controls");
        ImGui::Text("Brush Settings");
        ImGui::ColorEdit3("Brush Color", brushColor);
        ImGui::SliderFloat("Brush Size", &brushSize, 0.1f, 5.0f);
        
        ImGui::Separator();

        // 2. Fluidity Controls
        ImGui::Text("Fluid Dynamics");
        if (ImGui::SliderFloat("Viscosity", &fluidViscosity, 0.0f, 1.0f)) {
            // You'll need to update your solver or make these variables public/global
        }
        ImGui::SliderFloat("Diffusion", &fluidDiffusion, 0.0f, 0.01f, "%.4f");

        if (ImGui::Button("Reset Canvas", ImVec2(120, 0))) {
            resetCanvas(paintFluid);
        }

        ImGui::SameLine();

        if (ImGui::Button("Save Image", ImVec2(120, 0))) {
            saveCanvas(paintFluid);
        }

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    //float xpos = static_cast<float>(xposIn);
    //float ypos = static_cast<float>(yposIn);

    //if (firstMouse)
    //{
    //    lastX = xpos;
    //    lastY = ypos;
    //    firstMouse = false;
    //}

    //float xoffset = xpos - lastX;
    //float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    //lastX = xpos;
    //lastY = ypos;

    //camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
