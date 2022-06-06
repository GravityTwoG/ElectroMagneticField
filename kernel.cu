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

__device__ const float starB = -5e-10; // Tsl
__device__ const float C = 3e8;      // m/s
__device__ const float lambda = 1;// m
__device__ const int4 d_magneticField = { 
	WINDOW_WIDTH/3, // x-start
	-WINDOW_HEIGHT*10,              // y-start
	WINDOW_WIDTH*10,   // x-end
	WINDOW_HEIGHT*10   // y-end
};

const float starV = 5e4; // m/s
const float V = starV / C;

/* charge constants */
__constant__ const float K = 1e20;
__constant__ float MIN_DISTANCE = 1.0f; // not to divide by zero

const float MIN_CHARGE = 0.2e-19;
const float MAX_CHARGE = 1.6e-19;
const char MAX_CHARGE_COUNT = 30;

char chargeCount = 0;
__constant__ char dev_chargeCount;

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

/* OpenGL interoperability */
dim3 blocks, threads;
GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource;

/* charge selection */
const int DETECT_CHARGE_RANGE = 10;
int selectedChargeIndex = -1;
bool isDragging = false;

static void cudaCheckError(cudaError_t err, const char* file, int line);
#define HANDLE_ERROR( err ) (cudaCheckError( err, __FILE__, __LINE__ ))

void createVBO(GLuint* vbo, cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, cudaGraphicsResource* vbo_res);

void onKeyEvent(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		printf("Exit application\n");

		glutLeaveMainLoop();
		break;
	}
}

__device__ float length(const float2& q) {
	return sqrtf(q.x * q.x + q.y * q.y);
}

__device__ float length2(const float2& q) {
	return q.x * q.x + q.y * q.y;
}

__device__ bool isInMagneticField(float x, float y) {
	if (x < d_magneticField.x) return false;
	if (x > d_magneticField.z) return false;
	if (y < d_magneticField.y) return false;
	if (y > d_magneticField.w) return false;

	return true;
}

// apply Columbus Law
__global__ void dev_applyMagneticField(uchar4* screen, Particle* dev_charges, float dt) {
	int charge_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (charge_i >= MAX_CHARGE_COUNT) return;

	Particle& particle = dev_charges[charge_i];
	if (!particle.isPhysical) return;

	float2 v = { particle.vx, particle.vy };

	particle.x += dt * v.x;
	particle.y += dt * v.y;
	if (isInMagneticField(particle.x, particle.y)) {
		float B = starB * particle.charge * lambda / (particle.mass * C);
		particle.vx +=  dt * B * v.y;
		particle.vy += -dt * B * v.x;
	}

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
		//pixel.y = 255;
	}
}

// apply Columbus Law
__global__ void dev_applyElectricField(uchar4* screen, Particle* dev_charges, float dt) {
	int charge_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (charge_i >= MAX_CHARGE_COUNT) return;

	Particle& currentParticle = dev_charges[charge_i];
	if (!currentParticle.isPhysical) return;

	float2 force = { 0.0f, 0.0f };
	// iterate over all paricles and compute resulted force vector
	for (char i = 0; i < dev_chargeCount; i++) {
		const Particle& particle = dev_charges[i];
		float2 t_force = {
			currentParticle.x - particle.x,
			currentParticle.y - particle.y
		};

		float lengthSquared = length2(t_force) + MIN_DISTANCE;
		float e = particle.charge / sqrt(lengthSquared * lengthSquared * lengthSquared);
		t_force.x *= e;
		t_force.y *= e;

		force.x += t_force.x;
		force.y += t_force.y;
	}

	const float localK = K * currentParticle.charge / currentParticle.mass / 1000.0;
	force.x = force.x * localK * currentParticle.vx;
	force.y = force.y * localK * currentParticle.vx;

	__syncthreads();

	currentParticle.vx += dt * currentParticle.x;
	currentParticle.vy += dt * currentParticle.y;
	currentParticle.x += dt * force.x;
	currentParticle.y += dt * force.y;

	if (currentParticle.x >= WINDOW_WIDTH) currentParticle.x = WINDOW_WIDTH - 1;
	if (currentParticle.x < 0) currentParticle.x = 0;
	if (currentParticle.y >= WINDOW_HEIGHT) currentParticle.y = WINDOW_HEIGHT - 1;
	if (currentParticle.y < 0) currentParticle.y = 0;
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
__global__ void dev_renderFrame(uchar4* screen, Particle* dev_charges) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return;

	float2 force;
	force.x = force.y = 0.0f;

	float E = 0;
	float2 t_force;
	// iterate over all charges and compute resulted force vector
	for (char i = 0; i < dev_chargeCount; i++) {
		const Particle& particle = dev_charges[i];
		t_force.x = x - particle.x; // dx
		t_force.y = y - particle.y; // dy

		// x^2 + y^2
		float lengthSquared = length2(t_force) + MIN_DISTANCE;

		//e = q / (x^2 + y^2)^(3/2)
		float e = particle.charge / sqrtf(lengthSquared * lengthSquared * lengthSquared);
		E += e;
		t_force.x *= e;
		t_force.y *= e;

		force.x += t_force.x;
		force.y += t_force.y;
	}

	force.x *= K;
	force.y *= K;

	// set color on current pixel
	uchar4& pixel = screen[x + y * WINDOW_WIDTH];
	//pixel.x = pixel.y = pixel.z = 0;
	pixel.w = 255;

	float l = length(force); // 
	if (l < 0.2) return;

	float maxL = 1.0;
	if (E > 0.0) {
		pixel.x = l > maxL ? 255 : l/maxL * 255;
	} else {
		pixel.z = l > maxL ? 255 : l/maxL * 255 ;
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

	// Kernel Time measure
	cudaEvent_t startEvent, stopEvent;
	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));
	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	// Compute Image
	dev_applyMagneticField<<<1, MAX_CHARGE_COUNT>>>(
		dev_screen, dev_charges, 
		elapsedTime / 1000.0 * C / lambda
	);
	//dev_applyElectricField<<<1, MAX_CHARGE_COUNT>>>(dev_screen, dev_charges, elapsedTime / 1.0);
	dev_renderFrame<<<blocks, threads>>>(dev_screen, dev_charges);
	HANDLE_ERROR(cudaDeviceSynchronize());


	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	// Kernel Time measure
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
	
	// Draw electric field
	glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glColor4f(0.4f, 0.4f, 1.0f, 0.2f);
	glRecti(
		magneticField.x, magneticField.y,
		magneticField.z, magneticField.w
	);

	// draw selected point
	glPointSize(3.0f);
	glColor3f(0.0f, 1.0f, 1.0f);
	glBegin(GL_POINTS);
		glVertex2i(
			charges[selectedChargeIndex].x, 
			charges[selectedChargeIndex].y
		);
	glEnd();

	glutSwapBuffers();
}

void addCharge(int x, int y) {
	HANDLE_ERROR(
		cudaMemcpy(charges, dev_charges, chargeCount * sizeof(Particle), cudaMemcpyDeviceToHost)
	);
	
	if (chargeCount < MAX_CHARGE_COUNT) {
		chargeCount++;
	} else {
		// remove first charge
		for (int i = 0; i < MAX_CHARGE_COUNT - 1; ++i) {
			charges[i] = charges[i + 1];
		}
	}

	float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
	float newCharge = MIN_CHARGE + (float)scale * (MAX_CHARGE - MIN_CHARGE);      /* [min, max] */

	if (scale <= 0.5) {
		newCharge = -newCharge;
	}
	
	float vScale = rand() / (float)RAND_MAX; /* [0, 1.0] */

	charges[chargeCount - 1].x = x;
	charges[chargeCount - 1].y = y;
	charges[chargeCount - 1].charge = newCharge;
	charges[chargeCount - 1].vx = V * vScale;
	charges[chargeCount - 1].vy = 0.0f;
	charges[chargeCount - 1].mass = fabs(newCharge / 10e10);
	charges[chargeCount - 1].isPhysical = true;

	printf(
		"Debug: Charge #%d (%.0f, %.0f, %.0f)\n", chargeCount - 1,
		charges[chargeCount - 1].x, charges[chargeCount - 1].y,
		charges[chargeCount - 1].charge
	);
	printf("Charges %d\n", chargeCount);

	HANDLE_ERROR(
		cudaMemcpy(dev_charges, charges, chargeCount * sizeof(Particle), cudaMemcpyHostToDevice)
	);
	HANDLE_ERROR(
		cudaMemcpyToSymbol(dev_chargeCount, &chargeCount, sizeof(chargeCount))
	);
}


void onMouseEvent(int button, int state, int x, int y) {
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) {
		chargeCount = 0;
		HANDLE_ERROR(
			cudaMemcpyToSymbol(dev_chargeCount, &chargeCount, sizeof(chargeCount))
		);
		uchar4* dev_screen;
		size_t size;

		HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
		HANDLE_ERROR(
			cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource)
		);
		dev_clearFrame<<<blocks, threads >>>(dev_screen);
		HANDLE_ERROR(cudaDeviceSynchronize());
		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
		glutPostRedisplay();
		return;
	}
	
	if (button != GLUT_LEFT_BUTTON) return;

	// Drag, start dragging
	/*if (state == GLUT_DOWN && selectedChargeIndex != -1) {
		isDragging = true;
		printf(
			"Drag particle #%d with charge %.2f... ", 
			selectedChargeIndex, 
			charges[selectedChargeIndex].charge
		);
		charges[selectedChargeIndex].isPhysical = false;
		HANDLE_ERROR(cudaMemcpy(
			dev_charges + selectedChargeIndex,
			charges + selectedChargeIndex,
			1 * sizeof(Particle),
			cudaMemcpyHostToDevice
		));
		return;
	}*/
	
	if (state == GLUT_UP) {
		if (selectedChargeIndex != -1) { // Drop, stop dragging
			isDragging = false;
			charges[selectedChargeIndex].isPhysical = true;
			HANDLE_ERROR(cudaMemcpy(
				dev_charges + selectedChargeIndex,
				charges + selectedChargeIndex,
				1 * sizeof(Particle),
				cudaMemcpyHostToDevice
			));
			printf("Drop\n");
		} else {
			addCharge(x, WINDOW_HEIGHT - y);
		}
	}
}

void onMouseMove(int x, int y) {
	if (isDragging && selectedChargeIndex != -1) {
		if (x >= WINDOW_WIDTH) {
			charges[selectedChargeIndex].x = WINDOW_WIDTH - 1;
		} else if (x < 0) {
			charges[selectedChargeIndex].x = 0;
		} else {
			charges[selectedChargeIndex].x = x;
		}

		if (y >= WINDOW_HEIGHT) {
			charges[selectedChargeIndex].y = 0;
		} else if (y < 0) {
			charges[selectedChargeIndex].y = WINDOW_HEIGHT - 1;
		} else {
			charges[selectedChargeIndex].y = WINDOW_HEIGHT - y;
		}

		HANDLE_ERROR(
			cudaMemcpy(
				dev_charges + selectedChargeIndex,
				charges + selectedChargeIndex,
				1 * sizeof(Particle),
				cudaMemcpyHostToDevice
		));
	}
}

// Detect selected charge
void mouseTrack(int x, int y) {
	if (isDragging) return;

	HANDLE_ERROR(
		cudaMemcpy(charges, dev_charges, chargeCount * sizeof(Particle), cudaMemcpyDeviceToHost)
	);

	for (int i = 0; i < chargeCount; i++) {
		int dx = x - charges[i].x;
		int dy = (WINDOW_HEIGHT - y) - charges[i].y;

		if (charges[i].x >= 10 * WINDOW_WIDTH) return;
		if (charges[i].x < -10 * WINDOW_WIDTH) return;
		if (charges[i].y >= 10 * WINDOW_HEIGHT) return;
		if (charges[i].y < -10 * WINDOW_HEIGHT) return;

		if (dx * dx + dy * dy < DETECT_CHARGE_RANGE * DETECT_CHARGE_RANGE) {
			selectedChargeIndex = i;
			printf("#%d, x: %f, y: %f\n", selectedChargeIndex, charges[i].x, charges[i].y);
			return;
		}
	}

	selectedChargeIndex = -1;
}

void initCuda(int deviceId) {
	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties, deviceId));

	threads.x = 32;
	// to avoid cudaErrorLaunchOutOfResources error
	threads.y = properties.maxThreadsPerBlock / threads.x - 2;

	blocks.x = (WINDOW_WIDTH + threads.x - 1) / threads.x;
	blocks.y = (WINDOW_HEIGHT + threads.y - 1) / threads.y;

	cudaMalloc(&dev_charges, sizeof(Particle) * MAX_CHARGE_COUNT);

	printf(
		"Debug: blocks(%d, %d), threads(%d, %d)\nCalculated Resolution: %d x %d\n",
		blocks.x, blocks.y, threads.x, threads.y, blocks.x * threads.x,
		blocks.y * threads.y
	);
}

void initGlut(int argc, char** argv) {
	// Initialize freeglut
	glutInit(&argc, argv);
	
	//glutInitDisplayMode(GLUT_RGBA);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	int posX = glutGet(GLUT_SCREEN_WIDTH) / 2 - WINDOW_WIDTH / 2;
	int posY = glutGet(GLUT_SCREEN_HEIGHT) / 2 - WINDOW_HEIGHT / 2;
	glutInitWindowPosition(posX, posY);
	glutCreateWindow("Lab-4");

	glutIdleFunc(idle);
	glutDisplayFunc(draw);
	glutKeyboardFunc(onKeyEvent);
	glutMouseFunc(onMouseEvent);
	glutMotionFunc(onMouseMove);
	glutPassiveMotionFunc(mouseTrack);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)WINDOW_WIDTH, 0.0, (GLdouble)WINDOW_HEIGHT);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	glewInit();
}

int main(int argc, char** argv) {
	srand(time(0));
	initCuda(0);
	initGlut(argc, argv);

	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	glutMainLoop();

	deleteVBO(&vbo, cuda_vbo_resource);

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