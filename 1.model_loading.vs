#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Particles are already in world space, but we apply 'model' 
    // in case you want to move the entire simulation.
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}