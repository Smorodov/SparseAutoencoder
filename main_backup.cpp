#include <iostream>
#include <vector>
#include <stdio.h>
//#include <Eigen/Dense>
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/core/eigen.hpp"
#include "fstream"
#include "iostream"
#include <time.h>
#include <stdlib.h>
#include <lbfgs.h>
using namespace std;
using namespace cv;
//using namespace Eigen;
const double E=2.7182818284590452353602874713527;
// Это сами патчи
Mat Patches;
//-----------------------------------------------------------------------------------------------------
// Сигмоид (поэлементный) матрицы
//-----------------------------------------------------------------------------------------------------
Mat sigmoid(Mat& x)
{
	Mat res(x.rows,x.cols,CV_64FC1);
	exp(-x,res);
	res=1.0/(res+1.0);
	return res;
}
/*
//-----------------------------------------------------------------------------------------------------
// Берем число из интервала [0,1) согласно равномерному распределению
//-----------------------------------------------------------------------------------------------------
double Rand(void)
{
	return double(rand())/double(RAND_MAX);
}
*/
//-----------------------------------------------------------------------------------------------------
// Функция потерь и градиенты
//-----------------------------------------------------------------------------------------------------
void sparseAutoencoderCost(Mat &W,int visibleSize,int hiddenSize,double lambda,double sparsityParam,double beta, Mat& data,double* cost,Mat& grad)
{
	// visibleSize: количество входных узлов (probably 64)
	// hiddenSize: количество скрытых узнов (probably 25)
	// lambda: коэффициент ослабления весов
	// sparsityParam: Желаемый уровень активации нейронов скрытого слоя (Ро).
	// beta: Коэффициент (вес) слагаемого отвечающего за разреженность.
	// data: Наша матрица 64x10000 содержащая обучающую выборку.
	// таким образом, data(:,i) это i-th обучающая пара (вход и выход, в данном случае одно и то-же).
	//
	// Входной параметр W это вектор (т.к. minFunc ожидает, что параметр является вектором).
	// Сначала мы разобъем W на куски (W1, W2, b1, b2), чтобы все было как в лекции.

	// Веса, соединяющие входной вектор и скрытый слой
	Mat W1=W(Range(0,hiddenSize*visibleSize),Range::all()).clone();
	W1 = W1.reshape(1,hiddenSize);
	// Веса, соединяющие скрытый слой и выход
	Mat W2=W(Range(hiddenSize*visibleSize,2*hiddenSize*visibleSize),Range::all()).clone();
	W2 = W2.reshape(1,visibleSize);
	// Смещения нейронов скрытого слоя
	Mat b1=W(Range(2*hiddenSize*visibleSize,2*hiddenSize*visibleSize+hiddenSize),Range::all()).clone();
	// Смещения нейронов выходного слоя
	Mat b2=W(Range(2*hiddenSize*visibleSize+hiddenSize,W.rows),Range::all()).clone();

	// Функция стоимости и градиенты (Ваш код должен рассчитать эти значения).
	// Тут все инициализируется нулями.
	Mat W1grad = Mat::zeros(W1.rows,W1.cols,CV_64FC1);
	Mat W2grad = Mat::zeros(W2.rows,W2.cols,CV_64FC1);
	Mat b1grad =  Mat::zeros(b1.rows,b1.cols,CV_64FC1);
	Mat b2grad =  Mat::zeros(b2.rows,b2.cols,CV_64FC1);

	double numPatches=data.cols;

	Mat avgActivations=Mat::zeros(W1.rows,1,CV_64FC1);
	static Mat storedHiddenValues = Mat::zeros(hiddenSize, numPatches,CV_64FC1);
	static Mat storedOutputValues = Mat::zeros(visibleSize, numPatches,CV_64FC1);

	double J=0;
	static Mat X,z2,z3,a2,a3,tmp;
	//----------------------------
	// прямой проход (расчет выхода сети)
	//----------------------------
	for (int i=0;i<numPatches;i++)
	{
		data.col(i).copyTo(X);
		z2=W1*X+b1;
		a2=sigmoid(z2);
		avgActivations=avgActivations+a2;
		z3=W2*a2+b2;
		a3=sigmoid(z3);
		// сохраним результаты прямого хода
		a2.copyTo(storedHiddenValues.col(i));
		a3.copyTo(storedOutputValues.col(i));
		// Слагаемое функции потерь (сумма квадратов ошибки)
		tmp=a3-X;
		pow(tmp,2,tmp);
		J=J+0.5*sum(tmp)[0];
	}
	//----------------------------
	// Вычисления, связанные с условием разреженности
	// то есть чтобы в скрытом слое на поданный сигнал
	// активировалось лишь небольшое количество нейронов
	//----------------------------

	// из известной суммы найдем среднее
	avgActivations=avgActivations/numPatches;

	// Добавляется к дельте скрытого слоя при обратном проходе
	Mat sparsity_grad=beta*(-(sparsityParam/avgActivations)+((1.0-sparsityParam)/(1.0-avgActivations)));

	// Слагаемые дивергенции Куллбэка-Лейблера
	//Mat tmp1;
	log(sparsityParam/avgActivations,tmp);
	Mat KL1=sparsityParam*tmp;
	log((1.0-sparsityParam)/(1.0-avgActivations),tmp);
	Mat KL2=(1.0-sparsityParam)*tmp;
	// дивергенция Куллбэка-Лейблера (сумма элементов по всей выборке данных)
	double KL_divergence=sum(KL1+KL2)[0];
	// Функция потерь (минимизируемый функционал)

	static Mat W1Sqr;
	cv::pow(W1,2.0,W1Sqr);
	static Mat W2Sqr;
	cv::pow(W2,2.0,W2Sqr);
	(*cost)=(J/numPatches)+lambda*0.5*(sum(W1Sqr)[0]+sum(W2Sqr)[0])+beta*KL_divergence;
	//----------------------------
	// обратное распространение ошибки
	//----------------------------

	for (int i=0;i<numPatches;i++)
	{
		data.col(i).copyTo(X);
		// достаем ранее сохраненные веса
		storedHiddenValues.col(i).copyTo(a2);
		storedOutputValues.col(i).copyTo(a3);

		// ошибка выходного слоя
		Mat delta_3=(a3-X).mul(a3.mul(1.0-a3));
		// ошибка скрытого слоя
		Mat delta_2=(W2.t()*delta_3+sparsity_grad).mul(a2.mul(1.0-a2));

		W1grad+=delta_2*X.t();
		W2grad+=delta_3*a2.t();

		b1grad+=delta_2;
		b2grad+=delta_3;
	}

	//----------------------------
	// Градиенты весов
	//----------------------------
	W1grad=W1grad/numPatches+(lambda)*W1;
	W2grad=W2grad/numPatches+(lambda)*W2;
	//----------------------------
	// Градиенты смещений
	//----------------------------
	b1grad = b1grad/numPatches;
	b2grad = b2grad/numPatches;
	//----------------------------
	// Соберем вычисленные значения
	// градиентов в вектор-столбец
	// (подходящий для minFunc).
	//----------------------------

	W1grad=W1grad.reshape(1,hiddenSize*visibleSize);
	W2grad=W2grad.reshape(1,hiddenSize*visibleSize);
	b1grad=b1grad.reshape(1,hiddenSize);
	b2grad=b2grad.reshape(1,visibleSize);

	grad=Mat(W1grad.rows+W2grad.rows+b1grad.rows+b2grad.rows,1,CV_64FC1);
	W1grad.copyTo(grad(Range(0,hiddenSize*visibleSize),Range::all()));
	W2grad.copyTo(grad(Range(hiddenSize*visibleSize,2*hiddenSize*visibleSize),Range::all()));
	b1grad.copyTo(grad(Range(2*hiddenSize*visibleSize,2*hiddenSize*visibleSize+hiddenSize),Range::all()));
	b2grad.copyTo(grad(Range(2*hiddenSize*visibleSize+hiddenSize,grad.rows),Range::all()));
}
//------------------------------------------------------------
// Функция инициализации параметров
//------------------------------------------------------------
Mat initializeParameters(int hiddenSize,int visibleSize)
{
	Mat theta;
	// Инициализируем веса связей случайными величинами, основываясь на размерах сети.
	double  r = sqrt(6.0 / ((double)hiddenSize+(double)visibleSize+1.0));
	// делаем выборку из интервала [-r, r] по равномерному закону распределения
	//W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
	Mat W1=Mat(hiddenSize,visibleSize,CV_64FC1);
	cv::randu(W1,-r,r);
	Mat W2=Mat(visibleSize,hiddenSize,CV_64FC1);
	cv::randu(W2,-r,r);
	//W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
	// смещения заполним нулями
	Mat b1 = Mat::zeros(hiddenSize, 1,CV_64FC1);
	Mat b2 = Mat::zeros(visibleSize, 1,CV_64FC1);
	// Переведем веса и смещения в форму вектора-столбца.
	// Этот шаг "разворачивает" все параметры (веса и смещения) в один вектор  
	// который может использоваться функцией minFunc. 

	//theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
	W1=W1.reshape(1,hiddenSize*visibleSize);
	W2=W2.reshape(1,hiddenSize*visibleSize);
	b1=b1.reshape(1,hiddenSize);
	b2=b2.reshape(1,visibleSize);
	theta=Mat(W1.rows+W2.rows+b1.rows+b2.rows,1,CV_64FC1);
	W1.copyTo(theta(Range(0,hiddenSize*visibleSize),Range::all()));
	W2.copyTo(theta(Range(hiddenSize*visibleSize,2*hiddenSize*visibleSize),Range::all()));
	b1.copyTo(theta(Range(2*hiddenSize*visibleSize,2*hiddenSize*visibleSize+hiddenSize),Range::all()));
	b2.copyTo(theta(Range(2*hiddenSize*visibleSize+hiddenSize,theta.rows),Range::all()));

	return theta;
}
//-----------------------------------------------------------------------------------------------------
// Рисовалка сети (скрытого слоя)
//-----------------------------------------------------------------------------------------------------
void DrawNet(Mat& net,Mat& patches_img,int hiddenSize,int visibleSize,int patch_side,int scale)
{
	// отобразим N_GridRows*N_GridCols первых патчей
	int N_GridRows=sqrtf(hiddenSize);
	int N_GridCols=N_GridRows;

	patches_img=Mat((patch_side+1)*N_GridRows*scale,(patch_side+1)*N_GridCols*scale,CV_64FC1);
	patches_img=0;
	Mat p=(net(Range(0,visibleSize*hiddenSize),Range::all())).t();

	p=p.reshape(1,hiddenSize).clone();
	normalize(p,p,0,1,cv::NORM_MINMAX); // приведем к удобному диапазону

	for(int i=0;i<N_GridRows*N_GridCols;i++)
	{	
		Mat p_im;
		p.row(i).copyTo(p_im);
		p_im=p_im.reshape(1,patch_side);

//		normalize(p_im,p_im,0,1,cv::NORM_MINMAX); // приведем к удобному диапазону

		for(int j=0;j<p_im.rows;j++)
		{
			for(int k=0;k<p_im.cols;k++)
			{
				Point pt1=Point((i/N_GridCols)*(patch_side+1)*scale+k*scale+0.5*scale,(i%N_GridCols)*(patch_side+1)*scale+j*scale+0.5*scale);
				Point pt2=Point(((i/N_GridCols)*(patch_side+1)+1)*scale+k*scale+0.5*scale,((i%N_GridCols)*(patch_side+1)+1)*scale+j*scale+0.5*scale);
				rectangle(patches_img,pt1,pt2,Scalar::all(p_im.at<double>(j,k)),-1);
			}
		}
	}
}

//-----------------------------------------------------------------------------------------------------
// ПАРАМЕТРЫ СЕТИ
//-----------------------------------------------------------------------------------------------------
int patch_side=8;
int visibleSize=patch_side*patch_side;
int hiddenSize=16;
Mat theta;
double lambda=0.0001;
double sparsityParam=0.04;
double beta=3;

//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
static lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x, // Вектор параметров
	lbfgsfloatval_t *g,		  // Вектор градиентов
	const int n,			  // Размерность вектора
	const lbfgsfloatval_t step
	)
{

	// *x - указывает равен theta.data, поэтому копирования не требуется

	Mat grad;// градиент
	lbfgsfloatval_t fx = 0.0; // Целевая (минимизируемая) функция

	sparseAutoencoderCost(theta,visibleSize,hiddenSize,lambda,sparsityParam,beta, Patches,&fx,grad);

	// забираем градиент
	memcpy(g,grad.data,n *sizeof(double));

	return fx;
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
static int progress(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
	)
{
	// *x - указывает равен theta.data, поэтому копирования не требуется
	// Рисуем его
	Mat patches_img;
	int scale=5; // степень увеличения
	DrawNet(theta,patches_img,hiddenSize,visibleSize,patch_side,scale);
	imshow("Patches",patches_img);
	waitKey(15);

	printf("Iteration %d:\n", k);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}

//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	// ----------------------------------------------
	// Инициализация генератора случайных чисел
	// ----------------------------------------------
	srand((unsigned)time(NULL));
	
	// Здесь будут изображения с которых будем брать патчи
	vector<Mat> Images;
	namedWindow("Image");
	namedWindow("Patches");
	// ----------------------------------------------
	// Грузим изображения, с которых будем брать патчи
	// ----------------------------------------------
	for(int i=0;i<10;i++)
	{
		stringstream str;
		str << "..\\images\\image"<<i<<".jpg";
		Mat img;
		imread(str.str(),0).convertTo(img,CV_64FC1,1.0/255.0);
		Images.push_back(img);
		cout << str.str() << endl;
	}
	// ----------------------------------------------
	// теперь набираем патчи
	// ----------------------------------------------
	double n_patches=10000;
	Patches=Mat(patch_side*patch_side,n_patches,CV_64FC1);
	RNG rng;
	for(int i=0;i<n_patches;i++)
	{
		int n_img=rng.uniform(0.0,1.0)*((double)Images.size()-1);
		int row=rng.uniform(0.0,1.0)*((double)Images[n_img].rows-patch_side);
		int col=rng.uniform(0.0,1.0)*((double)Images[n_img].cols-patch_side);
		Mat patch=Images[n_img](Range(row,row+patch_side),Range(col,col+patch_side)).clone();
		patch=patch.reshape(1,patch_side*patch_side).clone();
		patch.copyTo(Patches.col(i));
	}
	// -------------------------------------------------------------------------------
	// Нормализация 
	// (выравниваем по центру,
	// отсекаем выбросы,
	// приводим к интервалу 0.1-0.9, т.к на выходе у нас сигмоидные функции активации)
	// -------------------------------------------------------------------------------
	Scalar mean;
	Scalar stddev;
	cv::meanStdDev(Patches,mean,stddev);
	//sqrt(stddev,stddev);
	Patches-=mean;
	for(int i=0;i<Patches.rows;i++)
	{
		for(int j=0;j<Patches.cols;j++)
		{
			if(Patches.at<double>(i,j)>3.0*stddev[0]){Patches.at<double>(i,j)=3.0*stddev[0];}
			if(Patches.at<double>(i,j)<-3.0*stddev[0]){Patches.at<double>(i,j)=-3.0*stddev[0];}
		}
	}
	normalize(Patches,Patches,0.1,0.9,cv::NORM_MINMAX); // приведем к удобному диапазону
	// -------------------------------------------------------------------------------
	// Инициализация вектора параметров небольшими случайными величинами
	// -------------------------------------------------------------------------------
	theta=initializeParameters(hiddenSize,visibleSize);

	int ret = 0;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x = lbfgs_malloc(theta.rows);
	lbfgs_parameter_t param;
	// Initialize the parameters for the L-BFGS optimization. 
	lbfgs_parameter_init(&param);
	//
	// Start the L-BFGS optimization; this will invoke the callback functions
	// evaluate() and progress() when necessary.
	//
	ret = lbfgs(theta.rows,(lbfgsfloatval_t *) theta.data, &fx, evaluate, progress, NULL, &param);

	// Report the result.
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);

	lbfgs_free(x);

	waitKey(0);

	return 0;
}