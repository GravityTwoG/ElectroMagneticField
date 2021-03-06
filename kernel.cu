#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


const GLint WINDOW_WIDTH = 820;
const GLint WINDOW_HEIGHT = 640;
const int4 magneticField = { WINDOW_WIDTH / 3, 0, WINDOW_WIDTH, WINDOW_HEIGHT };

__device__ const float starB = 5e-6; // Tsl
__device__ const float2 starE = { 0, 1e-1 }; // V/m
__device__ const float lambda = 1e-3;// m
__device__ const float C = 3e8;      // m/s
__device__ const int4 dev_magneticField = { 
	WINDOW_WIDTH/3, // x-start
	-WINDOW_HEIGHT*10,              // y-start
	WINDOW_WIDTH*10,   // x-end
	WINDOW_HEIGHT*10   // y-end
};

const float TIME_SCALE = 1e-4;
const float starV = 5e4; // m/s
const float V_MAX = starV / C;
const float V_MIN = 0.3 * V_MAX;

const float MAX_CHARGE = 1.6e-19;
const float MIN_CHARGE = 0.3 * MAX_CHARGE;
__constant__ const float K = 1.5e21;

const int MAX_CHARGE_COUNT = 20;

struct Particle {
	float x;
	float y;
	float vx;
	float vy;
	float charge;
	float mass;
	bool isPhysical;
};

Particle charges[MAX_CHARGE_COUNT]; 
Particle* dev_charges;
int chargeCount = 0;
__device__ int* dev_chargeCount;


dim3 blocks, threads;
/* OpenGL interoperability */
GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource;

static void cudaCheckError(cudaError_t err, const char* file, int line);
#define HANDLE_ERROR( err ) (cudaCheckError( err, __FILE__, __LINE__ ))

void createVBO(GLuint* vbo, cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, cudaGraphicsResource* vbo_res);


__device__ bool isInMagneticField(float x, float y) {
	if (x < dev_magneticField.x) return false;
	if (x > dev_magneticField.z) return false;
	if (y < dev_magneticField.y) return false;
	if (y > dev_magneticField.w) return false;

	return true;
}

__device__ inline float4 dF(const Particle& p) {
	if (isInMagneticField(p.x, p.y)) {
		float k = p.charge * lambda / (p.mass * C);
		float B = starB * k;
		float2 E = { 
			starE.x * k / C, 
			starE.y * k / C
		};

		return {
			p.vx,
			p.vy,
			E.x + B * p.vy,
			E.y - B * p.vx
		};
	}

	return {
		p.vx,
		p.vy,
		0,
		0
	};
}

__global__ void dev_applyMagneticField(
	uchar4* screen, Particle* dev_charges, int* d_chargesCount, float dt
) {
	int charge_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (charge_i >= *d_chargesCount) return;

	Particle& particle = dev_charges[charge_i];
	if (!particle.isPhysical) return;

	//float4 fi = dF(particle);
	//particle.x += dt * fi.x; // x + dx
	//particle.y += dt * fi.y; // y + dy
	//if (isInMagneticField(particle.x, particle.y)) {
	//	particle.vx += dt * fi.z;
	//	particle.vy += dt * fi.w;
	//}

	Particle p2 = particle;
	float4 d1 = dF(p2);

	p2.x = particle.x + dt * d1.x / 2;
	p2.y = particle.y + dt * d1.y / 2;
	p2.vx = particle.vx + dt * d1.z / 2;
	p2.vy = particle.vy + dt * d1.w / 2;
	float4 d2 = dF(p2);

	p2.x = particle.x + dt * d2.x / 2;
	p2.y = particle.y + dt * d2.y / 2;
	p2.vx = particle.vx + dt * d2.z / 2;
	p2.vy = particle.vy + dt * d2.w / 2;
	float4 d3 = dF(p2);
	
	p2.x = particle.x + dt * d3.x;
	p2.y = particle.y + dt * d3.y;
	p2.vx = particle.vx + dt * d3.z;
	p2.vy = particle.vy + dt * d3.w;
	float4 d4 = dF(p2);

	particle.x += dt / 6 * (d1.x + 2*d2.x + 2*d3.x + d4.x); // x + dx
	particle.y += dt / 6 * (d1.y + 2*d2.y + 2*d3.y + d4.y); // y + dy
	particle.vx += dt / 6 * (d1.z + 2 * d2.z + 2 * d3.z + d4.z);
	particle.vy += dt / 6 * (d1.w + 2 * d2.w + 2 * d3.w + d4.w);

	if (particle.x >= 10 * WINDOW_WIDTH) particle.isPhysical = false;
	if (particle.x < -10 * WINDOW_WIDTH) particle.isPhysical = false;
	if (particle.y >= 10 * WINDOW_HEIGHT) particle.isPhysical = false;
	if (particle.y < -10 * WINDOW_HEIGHT) particle.isPhysical = false;

	if (
		particle.x < WINDOW_WIDTH && 
		particle.x >=  0 && 
		particle.y < WINDOW_HEIGHT && 
		particle.y >= 0
	) {
		uchar4& pixel = screen[(int)particle.x + (int)particle.y * WINDOW_WIDTH];
		pixel.y = 250;
	}
}

__global__ void dev_clearFrame(uchar4* screen) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return;

	uchar4& pixel = screen[x + y * WINDOW_WIDTH];
	pixel.x = 0;
	pixel.y = 0;
	pixel.z = 0;
	pixel.w = 255;
}

// Compute electric field
__global__ void dev_renderFrame(uchar4* screen, Particle* dev_charges, int* d_chargesCount) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return;

	float E = 0;
	for (char i = 0; i < *d_chargesCount; i++) {
		const Particle& particle = dev_charges[i];
		float2 t_force = {
			x - particle.x, // dx
			y - particle.y  // dy
		};

		float length2 = t_force.x*t_force.x + t_force.y*t_force.y + 1;
		E += particle.charge / length2;
	}

	uchar4& pixel = screen[x + y * WINDOW_WIDTH];
	pixel.y = 0;
	pixel.w = 255;

	int brightness = K * fabs(E);
	if (E > 0.0) {
		pixel.x = pixel.x > brightness 
			? pixel.x 
			: brightness;
	} else {
		pixel.z = pixel.z > brightness 
			? pixel.z 
			: brightness;
	}
}

float elapsedTime = 0.0f;

void idle(void) {
	uchar4* dev_screen;
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource)
	);

	cudaEvent_t startEvent, stopEvent;
	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));
	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	//float elapsedTimeS = elapsedTime / 1000.0;
	float elapsedTimeS = 1 / 1000.0;
	float dtau = elapsedTimeS * C / lambda;
	dev_renderFrame<<<blocks, threads>>>(dev_screen, dev_charges, dev_chargeCount);
	dev_applyMagneticField<<<1, MAX_CHARGE_COUNT>>>(
		dev_screen, dev_charges, dev_chargeCount,
		dtau * TIME_SCALE
	);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	HANDLE_ERROR(cudaEventSynchronize(stopEvent));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	char fps[256];
	sprintf(fps, "%3.2f ms per frame (FPS: %3.1f)", elapsedTime,
		1000 / elapsedTime);
	glutSetWindowTitle(fps);

	glutPostRedisplay();
}

void draw(void) {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	
	glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glColor4f(0.4f, 0.4f, 1.0f, 0.2f);
	glRecti(
		magneticField.x, magneticField.y,
		magneticField.z, magneticField.w
	);

	glutSwapBuffers();
}

void clearScreen() {
	chargeCount = 0;
	HANDLE_ERROR(
		cudaMemcpy(dev_chargeCount, &chargeCount, sizeof(chargeCount), cudaMemcpyHostToDevice)
	);
	uchar4* dev_screen;
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource)
	);
	dev_clearFrame<<<blocks, threads>>>(dev_screen);
	HANDLE_ERROR(cudaDeviceSynchronize());
	
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	glutPostRedisplay();
}

void addCharge(int x, int y) {
	if (chargeCount < MAX_CHARGE_COUNT) {
		chargeCount++;
	} else {
		for (int i = 0; i < MAX_CHARGE_COUNT - 1; ++i) {
			charges[i] = charges[i + 1];
		}
	}

	float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
	float newCharge = MIN_CHARGE + (float)scale * (MAX_CHARGE - MIN_CHARGE);      /* [min, max] */

	float scale2 = rand() / (float)RAND_MAX; /* [0, 1.0] */
	if (scale2 < 0.5) {
		newCharge = -newCharge;
	}
	
	float vScale = rand() / (float)RAND_MAX; /* [0, 1.0] */

	charges[chargeCount - 1].x = x;
	charges[chargeCount - 1].y = y;
	charges[chargeCount - 1].charge = newCharge;
	charges[chargeCount - 1].vx = V_MIN + vScale * (V_MAX - V_MIN);
	charges[chargeCount - 1].vy = 0.0f;
	charges[chargeCount - 1].mass = 9.11e-31;
	charges[chargeCount - 1].isPhysical = true;
}

void addCharges(int x, int y, int n) {
	HANDLE_ERROR(
		cudaMemcpy(charges, dev_charges, chargeCount * sizeof(Particle), cudaMemcpyDeviceToHost)
	);
	
	float disp = 40;
	for (int i = 0; i < MAX_CHARGE_COUNT && i < n; i++) {
		float dx = rand() / (float)RAND_MAX * disp;
		float dy = rand() / (float)RAND_MAX * disp;
		
		addCharge(x + dx - disp/2, y + dy - disp / 2);
		printf("Charges %d\n", chargeCount);
	}

	HANDLE_ERROR(
		cudaMemcpy(dev_charges, charges, chargeCount * sizeof(Particle), cudaMemcpyHostToDevice)
	);
	HANDLE_ERROR(
		cudaMemcpy(dev_chargeCount, &chargeCount, sizeof(chargeCount), cudaMemcpyHostToDevice)
	);
}


void onMouseEvent(int button, int state, int x, int y) {
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) {
		clearScreen();
		return;
	}
	
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
		addCharges(x, WINDOW_HEIGHT - y, 1);
		return;
	}

	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
		clearScreen();
		addCharges(x, WINDOW_HEIGHT - y, MAX_CHARGE_COUNT);
		return;
	}
}

void onResize(int width, int height) {
	glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
}

void initGlut(int argc, char** argv) {
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	int posX = glutGet(GLUT_SCREEN_WIDTH) / 2 - WINDOW_WIDTH / 2;
	int posY = glutGet(GLUT_SCREEN_HEIGHT) / 2 - WINDOW_HEIGHT / 2;
	glutInitWindowPosition(posX, posY);
	glutCreateWindow("Lab-4");

	glutIdleFunc(idle);
	glutDisplayFunc(draw);
	glutMouseFunc(onMouseEvent);
	glutReshapeFunc(onResize);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, WINDOW_WIDTH, 0.0, WINDOW_HEIGHT);

	// enable alpha-channel
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	glewInit();
}

int main(int argc, char** argv) {
	srand(time(0));
	initGlut(argc, argv);

	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties, 0));

	threads.x = 32;
	threads.y = properties.maxThreadsPerBlock / threads.x;

	blocks.x = (WINDOW_WIDTH + threads.x) / threads.x;
	blocks.y = (WINDOW_HEIGHT + threads.y) / threads.y;
	cudaMalloc((void**)&dev_charges, sizeof(Particle) * MAX_CHARGE_COUNT);
	cudaMalloc((void**)&dev_chargeCount, sizeof(chargeCount));
	HANDLE_ERROR(
		cudaMemcpy(dev_chargeCount, &chargeCount, sizeof(chargeCount), cudaMemcpyHostToDevice)
	);
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	glutMainLoop();

	deleteVBO(&vbo, cuda_vbo_resource);
	cudaFree(dev_charges);
	cudaFree(dev_chargeCount);

	return 0;
}

static void cudaCheckError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

void createVBO(
	GLuint* vbo,
	struct cudaGraphicsResource** vbo_res,
	unsigned int vbo_res_flags
) {
	unsigned int size = WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uchar4);

	glGenBuffers(1, vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res) {
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}