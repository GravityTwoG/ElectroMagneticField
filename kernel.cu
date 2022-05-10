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
__constant__ const float maxSolidColorLength = 1.0f;

const int MAX_CHARGE = 100;
const int MIN_CHARGE = -100;

const char MAX_CHARGE_COUNT = 30;
char chargeCount = 0;
__constant__ char dev_chargeCount;

float3 charges[MAX_CHARGE_COUNT]; // x, y, z == m
__constant__ float3 dev_charges[MAX_CHARGE_COUNT]; // x, y, z == m

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
	return (q.x * q.x + q.y * q.y);
}


__global__ void dev_renderFrame(uchar4* screen) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) return;

	float2 force, t_force;
	force.x = force.y = 0.0f;

	float E = 0;
	// iterate over all charges and compute resulted force vector
	for (char i = 0; i < dev_chargeCount; i++) {
		const float3& charge = dev_charges[i];
		float2& f = t_force;

		f.x = x - charge.x;
		f.y = y - charge.y;

		float l = length2(f) + MIN_DISTANCE;

		// 
		float e = charge.z / sqrt(l * l * l);
		E += e;
		f.x *= e;
		f.y *= e;

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
		pixel.x = (l > maxSolidColorLength ? 255 : l * 256 / maxSolidColorLength);
	} else {
		pixel.z = (l > maxSolidColorLength ? 255 : l * 256 / maxSolidColorLength);
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
	charges[chargeCount - 1].z = MIN_CHARGE + rand() % (MAX_CHARGE - MIN_CHARGE);

	printf(
		"Debug: Charge #%d (%.0f, %.0f, %.0f)\n", chargeCount - 1,
		charges[chargeCount - 1].x, charges[chargeCount - 1].y,
		charges[chargeCount - 1].z
	);

	HANDLE_ERROR(
		cudaMemcpyToSymbol(dev_charges, charges, chargeCount * sizeof(float3))
	);
	HANDLE_ERROR(
		cudaMemcpyToSymbol(dev_chargeCount, &chargeCount, sizeof(chargeCount))
	);
	printf("Charges %d\n", chargeCount);
	printf("Charge: %f\n", charges[chargeCount - 1].z);
}


void onMouseEvent(int button, int state, int x, int y) {
	if (button != GLUT_LEFT_BUTTON) return;

	// Drag, start dragging
	if (state == GLUT_DOWN && selectedChargeIndex != -1) {
		isDragging = true;
		printf("Drag charge #%d... ", selectedChargeIndex);
		return;
	}
	
	if (state == GLUT_UP) {
		if (selectedChargeIndex != -1) { // Drop, stop dragging
			isDragging = false;
			printf("Drop\n");
		} else {
			addCharge(x, WINDOW_HEIGHT - y);
		}
	}
}

void onMouseMove(int x, int y) {
	if (isDragging && selectedChargeIndex != -1) {
		printf(" drag... \n");
		charges[selectedChargeIndex].x = x;
		charges[selectedChargeIndex].y = WINDOW_HEIGHT - y;

		HANDLE_ERROR(
			cudaMemcpyToSymbol(dev_charges, charges, chargeCount * sizeof(float3))
		);
	}
}

// Detect selected charge
void mouseTrack(int x, int y) {
	if (isDragging) return;

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
	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

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