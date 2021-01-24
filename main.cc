#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif


#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <GL/glew.h>
#include <GL/glut.h>

#include "opengl.hh"
#include "vector.hh"

using clock_type = std::chrono::high_resolution_clock;
using float_duration = std::chrono::duration<float>;
using vec2 = Vector<float,2>;

// Original code: https://github.com/cerrno/mueller-sph
constexpr const float kernel_radius = 16;
constexpr const float particle_mass = 65;
constexpr const float poly6 = 315.f/(65.f*float(M_PI)*std::pow(kernel_radius,9));
constexpr const float spiky_grad = -45.f/(float(M_PI)*std::pow(kernel_radius,6));
constexpr const float visc_laplacian = 45.f/(float(M_PI)*std::pow(kernel_radius,6));
constexpr const float gas_const = 2000.f;
constexpr const float rest_density = 1000.f;
constexpr const float visc_const = 250.f;
constexpr const vec2 G(0.f, 12000*-9.8f);

struct Particle {

    vec2 position;
    vec2 velocity;
    vec2 force;
    float density;
    float pressure;

    Particle() = default;
    inline explicit Particle(vec2 x): position(x) {}

};

std::vector<Particle> particles;

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

const std::string src = R"(

kernel void compute_density_and_pressure(float poly6, float gas_const, float rest_density, float kernel_radius, float particle_mass,
 global float2 * positions, global float * pressures, global float * densities,) {

    float kernel_radius_squared = kernel_radius * kernel_radius;
    float sum = 0;
    int i = get_global_id(0);
    int nx = get_global_size(0);
    for (int j=0; j < nx; ++j){
        float sd = pow(length(positions[j] - positions[i]),2);
            if (sd < kernel_radius_squared) {
                sum += particle_mass * poly6 * pow(kernel_radius_squared - sd, 3);
            }
        }
    densities[i] = sum;
    pressures[i] = gas_const * (densities[i] - rest_density);
    
}

kernel void compute_forces(float visc_laplacian, float spiky_grad, float visc_const, float kernel_radius, float particle_mass
  global float2 * positions, global float2 * velocities, global float2 * forces, global float * pressures, global float * densities  ) {

    float2 G = {0.f, 12000*-9.8f};
    int i = get_global_id(0);
    int nx = get_global_size(0);
    float2 pressure_force = {0.f, 0.f};
    float2 viscosity_force  = {0.f, 0.f};
    for (int j=0; j<nx; ++j){
        if (j == i) { continue; }
        float2 delta = positions[j] - positions[i];
        float r = length(delta);
        if (r < kernel_radius) {
            pressure_force += -normalize(delta) * particle_mass * (pressures[i] + pressures[j])
                                  / (2.f * densities[j])
                                  * spiky_grad * pow(kernel_radius-r, 2.f);
            viscosity_force += visc_const * particle_mass * (velocities[j] - velocities[i])
                                   / densities[j] * visc_laplacian * (kernel_radius - r);
        }
    }
    float2 gravity_force = G * densities[i];
    forces[i] = (pressure_force + viscosity_force + gravity_force);             
 
}

kernel void compute_positions(int window_width, int window_height, float rest_density, float kernel_radius, float particle_mass,
 global float2 * positions, global float2 * velocities, global float2 * forces, global float * densities  ) {
        
    float rest_density = 1000.f;
    const float time_step = 0.0008f;
    const float eps = kernel_radius;
    const float damping = -0.5f;
    int i = get_global_id(0);
    velocities[i] += time_step * forces[i] / densities[i];
    positions[i] += time_step * velocities[i];
    // enforce boundary conditions
    if (positions[i].x - eps < 0.0f) {
        velocities[i].x *= damping;
        positions[i].x = eps;
    }
    if (positions[i].x + eps > window_width) {
        velocities[i].x *= damping;
        positions[i].x = window_width - eps;
    }
    if (positions[i].y - eps < 0.0f) {
        velocities[i].y *= damping;
        positions[i].y = eps;
    }
    if (positions[i].y +eps > window_height) {
        velocities[i].y *= damping;
        positions[i].y = window_height - eps;
    }
}


)";

OpenCL init_opencl(){
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
           throw;
        }

        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        
    cl::CommandQueue queue(context, device);
    return OpenCL{platform, device, context, program, queue};

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
       throw;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
       throw;
    }
    return OpenCL{}; 
}




void generate_particles() {
    std::random_device dev;
    std::default_random_engine prng(dev());
    float jitter = 1;
    std::uniform_real_distribution<float> dist_x(-jitter,jitter);
    std::uniform_real_distribution<float> dist_y(-jitter,jitter);
    int ni = 15;
    int nj = 40;
    float x0 = window_width*0.25f;
    float x1 = window_width*0.75f;
    float y0 = window_height*0.20f;
    float y1 = window_height*1.00f;
    float step = 1.5f*kernel_radius;
    for (float x=x0; x<x1; x+=step) {
        for (float y=y0; y<y1; y+=step) {
            particles.emplace_back(vec2{x+dist_x(prng),y+dist_y(prng)});
        }
    }
    std::clog << "No. of particles: " << particles.size() << std::endl;

}

void compute_density_and_pressure() {
    const auto kernel_radius_squared = kernel_radius*kernel_radius;
    #pragma omp parallel for schedule(dynamic)
    for (auto& a : particles) {
        float sum = 0;
        for (auto& b : particles) {
            auto sd = square(b.position-a.position);
            if (sd < kernel_radius_squared) {
                sum += particle_mass*poly6*std::pow(kernel_radius_squared-sd, 3);
            }
        }
        a.density = sum;
        a.pressure = gas_const*(a.density - rest_density);
    }
}

void compute_forces() {
    #pragma omp parallel for schedule(dynamic)
    for (auto& a : particles) {
        vec2 pressure_force(0.f, 0.f);
        vec2 viscosity_force(0.f, 0.f);
        for (auto& b : particles) {
            if (&a == &b) { continue; }
            auto delta = b.position - a.position;
            auto r = length(delta);
            if (r < kernel_radius) {
                pressure_force += -unit(delta)*particle_mass*(a.pressure + b.pressure)
                    / (2.f * b.density)
                    * spiky_grad*std::pow(kernel_radius-r,2.f);
                viscosity_force += visc_const*particle_mass*(b.velocity - a.velocity)
                    / b.density * visc_laplacian*(kernel_radius-r);
            }
        }
        vec2 gravity_force = G * a.density;
        a.force = pressure_force + viscosity_force + gravity_force;
    }
}

void compute_positions() {
    const float time_step = 0.0008f;
    const float eps = kernel_radius;
    const float damping = -0.5f;
    #pragma omp parallel for
    for (auto& p : particles) {
        // forward Euler integration
        p.velocity += time_step*p.force/p.density;
        p.position += time_step*p.velocity;
        // enforce boundary conditions
        if (p.position(0)-eps < 0.0f) {
            p.velocity(0) *= damping;
            p.position(0) = eps;
        }
        if (p.position(0)+eps > window_width) {
            p.velocity(0) *= damping;
            p.position(0) = window_width-eps;
        }
        if (p.position(1)-eps < 0.0f) {
            p.velocity(1) *= damping;
            p.position(1) = eps;
        }
        if (p.position(1)+eps > window_height) {
            p.velocity(1) *= damping;
            p.position(1) = window_height-eps;
        }
    }
}

void on_display() {
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,fbo); }
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    gluOrtho2D(0, window_width, 0, window_height);
    glColor4f(0.2f, 0.6f, 1.0f, 1);
    glBegin(GL_POINTS);
    for (const auto& particle : particles) {
        glVertex2f(particle.position(0), particle.position(1));
    }
    glEnd();
    glutSwapBuffers();
    if (no_screen) { glReadBuffer(GL_RENDERBUFFER); }
    recorder.record_frame();
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,0); }
}

void on_idle_cpu() {
    if (particles.empty()) { generate_particles(); }
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    auto t0 = clock_type::now();
    compute_density_and_pressure();
    compute_forces();
    compute_positions();
    auto t1 = clock_type::now();
    auto dt = duration_cast<float_duration>(t1-t0).count();
    std::clog
        << std::setw(20) << dt
        << std::setw(20) << 1.f/dt
        << std::endl;
	glutPostRedisplay();
}



OpenCL opencl = init_opencl();

cl::Kernel kernel_density_and_pressure(opencl.program, "compute_density_and_pressure");

cl::Kernel kernel_forces(opencl.program, "compute_forces");

cl::Kernel kernel_positions(opencl.program, "compute_positions");


struct opencl_vector{
    float x;
    float y;
};

void on_idle_gpu() {
    //std::clog << "GPU version is not implemented!" << std::endl; std::exit(1);
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;

    if (particles.empty()) {
        generate_particles();
        //Add constats to Args

        kernel_density_and_pressure.setArg(0, poly6);
        kernel_density_and_pressure.setArg(1, gas_const);
        kernel_density_and_pressure.setArg(2, rest_density);
        kernel_density_and_pressure.setArg(3, kernel_radius);
        kernel_density_and_pressure.setArg(4, particle_mass);

        kernel_forces.setArg(0, visc_laplacian);
        kernel_forces.setArg(1, spiky_grad);
        kernel_forces.setArg(2, visc_const);
        kernel_forces.setArg(3, kernel_radius);
        kernel_forces.setArg(4, particle_mass);

        kernel_positions.setArg(0, window_width);
        kernel_positions.setArg(1, window_height);
        kernel_positions.setArg(2, rest_density);
        kernel_positions.setArg(3, kernel_radius);
        kernel_positions.setArg(4, particle_mass);

    }

    int num_particles = particles.size();

    std::vector<float> densities, pressures;
    std::vector<opencl_vector> positions, velocities, forces;

    // Создаем векторы для передачи
    for (auto p : particles){

        densities.push_back((float) (p.density));
        pressures.push_back((float) (p.pressure));

        positions.push_back(opencl_vector{ p.position(0),p.position(1) });
        velocities.push_back(opencl_vector{ p.velocity(0),p.velocity(1) });
        forces.push_back(opencl_vector{ p.force(0), p.force(1) });
  
    }

    auto t0 = clock_type::now();
    // TODO see on_idle_cpu

    cl::Buffer d_pressures(opencl.context, begin(pressures), end(pressures), true);
    cl::Buffer d_densities(opencl.context, begin(densities), end(densities), true);  

    cl::Buffer d_positions(opencl.context, begin(positions), end(positions), true);
    cl::Buffer d_velocities(opencl.context, begin(velocities), end(velocities), true);
    cl::Buffer d_forces(opencl.context, begin(forces), end(forces), true);
    
    
    kernel_density_and_pressure.setArg(5, d_positions);
    kernel_density_and_pressure.setArg(6, d_pressures);
    kernel_density_and_pressure.setArg(7, d_densities);
    
    
    kernel_forces.setArg(5, d_positions);
    kernel_forces.setArg(6, d_velocities);
    kernel_forces.setArg(7, d_forces);
    kernel_forces.setArg(8, d_pressures);
    kernel_forces.setArg(9, d_densities);


    kernel_positions.setArg(5, d_positions);
    kernel_positions.setArg(6, d_velocities);
    kernel_positions.setArg(7, d_forces);
    kernel_positions.setArg(8, d_densities);

    opencl.queue.flush();
    try{
        opencl.queue.enqueueNDRangeKernel(kernel_density_and_pressure,cl::NullRange, cl::NDRange(num_particles),cl::NullRange);
    }
    catch (cl::Error err){
        printf("Error code is %i\n", err);
    };
    try{
        opencl.queue.enqueueNDRangeKernel(kernel_forces,cl::NullRange, cl::NDRange(num_particles),cl::NullRange);
    }
    catch (cl::Error err){
        printf("Error code is %i\n", err);
    };
    try{
        opencl.queue.enqueueNDRangeKernel(kernel_positions,cl::NullRange, cl::NDRange(num_particles),cl::NullRange);
    }
    catch (cl::Error err){
        printf("Error code is %i\n", err);
    };

    opencl.queue.finish();

    cl::copy(opencl.queue, d_pressures, begin(pressures), end(pressures));
    cl::copy(opencl.queue, d_densities, begin(densities), end(densities));

    cl::copy(opencl.queue, d_positions, begin(positions), end(positions));
    cl::copy(opencl.queue, d_velocities, begin(velocities), end(velocities));
    cl::copy(opencl.queue, d_forces, begin(forces), end(forces));


    opencl.queue.flush();

    for (int i=0; i<num_particles;++i) {
        particles[i].position[0] = positions[i].x;
        particles[i].position[1] = positions[i].y;
        particles[i].velocity[0] = velocities[i].x;
        particles[i].velocity[1] = velocities[i].y;
        particles[i].force[0] = forces[i].x;
        particles[i].force[1] = forces[i].y;
        particles[i].density = densities[i];
        particles[i].pressure = pressures[i];
    }

    auto t1 = clock_type::now();
    auto dt = duration_cast<float_duration>(t1-t0).count();
    std::clog
        << std::setw(20) << dt
        << std::setw(20) << 1.f/dt
        << std::endl;
	glutPostRedisplay();
}

void on_keyboard(unsigned char c, int x, int y) {
    switch(c) {
        case ' ':
            generate_particles();
            break;
        case 'r':
        case 'R':
            particles.clear();
            generate_particles();
            break;
    }
}

void print_column_names() {
    std::clog << std::setw(20) << "Frame duration";
    std::clog << std::setw(20) << "Frames per second";
    std::clog << '\n';
}

int main(int argc, char* argv[]) {
    enum class Version { CPU, GPU };
    Version version = Version::CPU;
    if (argc == 2) {
        std::string str(argv[1]);
        for (auto& ch : str) { ch = std::tolower(ch); }
        if (str == "gpu") { version = Version::GPU; }
    }
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(window_width, window_height);
	glutInit(&argc, argv);
	glutCreateWindow("SPH");
	glutDisplayFunc(on_display);
    glutReshapeFunc(on_reshape);
 
    switch (version) {
        case Version::CPU: glutIdleFunc(on_idle_cpu); break;
        case Version::GPU: glutIdleFunc(on_idle_gpu); break;
        default: return 1;
    }
	glutKeyboardFunc(on_keyboard);
    glewInit();
	init_opengl(kernel_radius);
    print_column_names();
	glutMainLoop();
    return 0;
}
