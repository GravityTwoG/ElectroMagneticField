#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


const GLint WINDOW_WIDTH = 640;
const GLint WINDOW_HEIGHT = 640;

/* charge constants */
__constant__ const float K = 20.0f;
__constant__ float MIN_DISTANCE = 0.1f; // not to divide by zero
__constant__ const float MAX_SOLID_COLOR = 1.0f;

const int MAX_CHARGE = 100;
const int MIN_CHARGE = -100;
const int MIN_CHARGE_ABS = 5;
const char MAX_CHARGE_COUNT = 30;

char chargeCount = 0;
__constant__ char dev_chargeCount;

struct Particle {
	float x;
	float y;
	float dx;
	float dy;
	float charge;
	float mass;
	bool isPhysical;
};

Particle charges[MAX_CHARGE_COUNT]; 
__constant__ Particle dev_charges[MAX_CHARGE_COUNT]; 

/* OpenGL interoperability */
dim3 blocks, threads;
GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource;

/* charge selection */
const int DETECT_CHARGE_RANGE = 20;
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

// apply ode
__global__ void dev_moveCharge(uchar4* screen) {
	int charge_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (charge_i >= MAX_CHARGE_COUNT) return;

	Particle& currentParticle = dev_charges[charge_i];

	if (!currentParticle.isPhysical) return;
	if (currentParticle.x > 10 * WINDOW_WIDTH) return;
	if (currentParticle.x < -10 * WINDOW_WIDTH) return;
	if (currentParticle.y > 10 * WINDOW_HEIGHT) return;
	if (currentParticle.y < -10 * WINDOW_HEIGHT) return;

	float2 force;
	force.x = force.y = 0.0f;

	// iterate over all charges and compute resulted force vector
	float2 t_force;
	for (char i = 0; i < dev_chargeCount; i++) {
		const Particle& particle = dev_charges[i];
		t_force.x = currentParticle.x - particle.x;
		t_force.y = currentParticle.y - particle.y;

		float l = length2(t_force) + MIN_DISTANCE;
		float e = particle.charge / sqrt(l * l * l);
		float maxE = 5000.0;
		if (e > maxE) e = maxE;
		if (e < -maxE) e = -maxE;
		t_force.x *= e;
		t_force.y *= e;

		force.x += t_force.x;
		force.y += t_force.y;
	}

	float localK = currentParticle.charge / currentParticle.mass;
	force.x *= localK;
	force.y *= localK;

	currentParticle.x += force.x;
	currentParticle.y += force.y;
}

__global__ void dev_renderFrame(uchar4* screen) {
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
	pixel.x = pixel.y = pixel.z = pixel.w = 0;

	float l = length(force); // 

	if (E >= 0.0) {
		pixel.x = (l > MAX_SOLID_COLOR ? 255 : l * 256 / MAX_SOLID_COLOR);
	} else {
		pixel.z = (l > MAX_SOLID_COLOR ? 255 : l * 256 / MAX_SOLID_COLOR);
	}
}


void idle(void) {
	uchar4* dev_screen;
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource)
	);

	// Kernel Time measure
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime = 0.0f;
	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));
	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	// Render Image
	dev_moveCharge<<<blocks, threads>>>(dev_screen);
	dev_renderFrame<<<blocks, threads>>>(dev_screen);
	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	// Kernel Time measure
	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	HANDLE_ERROR(cudaEventSynchronize(stopEvent));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	char fps[256];
	sprintf(fps, "Electric field: %3.2f ms per frame (FPS: %3.1f)", elapsedTime,
		1000 / elapsedTime);
	glutSetWindowTitle(fps);

	glutPostRedisplay();
}

void draw(void) {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

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
	if (chargeCount < MAX_CHARGE_COUNT) {
		chargeCount++;
	} else {
		// remove first charge
		for (int i = 0; i < MAX_CHARGE_COUNT - 1; ++i) {
			charges[i] = charges[i + 1];
		}
	}

	charges[chargeCount - 1].x = x;
	charges[chargeCount - 1].y = y;
	float newCharge = MIN_CHARGE + rand() % (MAX_CHARGE - MIN_CHARGE);
	if (newCharge >= 0 && newCharge < MIN_CHARGE_ABS) {
		newCharge = MIN_CHARGE_ABS;
	} else if (newCharge < 0 && newCharge > MIN_CHARGE_ABS) {
		newCharge = -MIN_CHARGE_ABS;
	}
	charges[chargeCount - 1].charge = newCharge;
	charges[chargeCount - 1].mass = fabs(newCharge / 10.0);
	charges[chargeCount - 1].isPhysical = true;

	printf(
		"Debug: Charge #%d (%.0f, %.0f, %.0f)\n", chargeCount - 1,
		charges[chargeCount - 1].x, charges[chargeCount - 1].y,
		charges[chargeCount - 1].charge
	);
	printf("Charges %d\n", chargeCount);

	HANDLE_ERROR(
		cudaMemcpyToSymbol(dev_charges, charges, chargeCount * sizeof(Particle))
	);
	HANDLE_ERROR(
		cudaMemcpyToSymbol(dev_chargeCount, &chargeCount, sizeof(chargeCount))
	);
}


void onMouseEvent(int button, int state, int x, int y) {
	if (button != GLUT_LEFT_BUTTON) return;

	// Drag, start dragging
	if (state == GLUT_DOWN && selectedChargeIndex != -1) {
		isDragging = true;
		printf(
			"Drag particle #%d with charge %.2f... ", 
			selectedChargeIndex, 
			charges[selectedChargeIndex].charge
		);
		charges[selectedChargeIndex].isPhysical = false;
		return;
	}
	
	if (state == GLUT_UP) {
		if (selectedChargeIndex != -1) { // Drop, stop dragging
			isDragging = false;
			charges[selectedChargeIndex].isPhysical = true;
			printf("Drop\n");
		} else {
			addCharge(x, WINDOW_HEIGHT - y);
		}
	}
}

void onMouseMove(int x, int y) {
	if (isDragging && selectedChargeIndex != -1) {
		//printf(" drag... \n");
		charges[selectedChargeIndex].x = x;
		charges[selectedChargeIndex].y = WINDOW_HEIGHT - y;

		HANDLE_ERROR(
			cudaMemcpyToSymbol(dev_charges, charges, chargeCount * sizeof(Particle))
		);
	}
}

// Detect selected charge
void mouseTrack(int x, int y) {
	if (isDragging) return;

	//HANDLE_ERROR(
	//	cudaMemcpy(charges, dev_charges, chargeCount * sizeof(Particle), cudaMemcpyDefault)
	//);

	int dx = 0, dy = 0;

	for (int i = 0; i < chargeCount; i++) {
		dx = x - charges[i].x;
		dy = (WINDOW_HEIGHT - y) - charges[i].y;

		if (dx * dx + dy * dy < DETECT_CHARGE_RANGE * DETECT_CHARGE_RANGE) {
			selectedChargeIndex = i;

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

	printf(
		"Debug: blocks(%d, %d), threads(%d, %d)\nCalculated Resolution: %d x %d\n",
		blocks.x, blocks.y, threads.x, threads.y, blocks.x * threads.x,
		blocks.y * threads.y
	);
}

void initGlut(int argc, char** argv) {
	// Initialize freeglut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	int posX = glutGet(GLUT_SCREEN_WIDTH) / 2 - WINDOW_WIDTH / 2;
	int posY = glutGet(GLUT_SCREEN_HEIGHT) / 2 - WINDOW_HEIGHT / 2;
	glutInitWindowPosition(posX, posY);
	glutCreateWindow("Electric field");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	glutIdleFunc(idle);
	glutKeyboardFunc(onKeyEvent);
	glutMouseFunc(onMouseEvent);
	glutMotionFunc(onMouseMove);
	glutPassiveMotionFunc(mouseTrack);
	glutDisplayFunc(draw);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)WINDOW_WIDTH, 0.0, (GLdouble)WINDOW_HEIGHT);

	glewInit();
}

int main(int argc, char** argv) {
	setbuf(stdout, NULL); // ?

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