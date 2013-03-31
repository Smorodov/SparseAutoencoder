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

const double E=2.7182818284590452353602874713527;
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
inline void endian_swap(unsigned short& x)
{
	x = (x>>8) | 
		(x<<8);
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
inline void endian_swap(unsigned int& x)
{
	x = (x>>24) | 
		((x<<8) & 0x00FF0000) |
		((x>>8) & 0x0000FF00) |
		(x<<24);
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
inline void endian_swap(unsigned __int64& x)
{
	x = (x>>56) | 
		((x<<40) & 0x00FF000000000000) |
		((x<<24) & 0x0000FF0000000000) |
		((x<<8)  & 0x000000FF00000000) |
		((x>>8)  & 0x00000000FF000000) |
		((x>>24) & 0x0000000000FF0000) |
		((x>>40) & 0x000000000000FF00) |
		(x<<56);
}
//-----------------------------------------------------------------------------------------------------
// Чтение меток NMIST (здесь не используется)
//-----------------------------------------------------------------------------------------------------
void read_mnist_labels(string fname,vector<unsigned char>& vec_lbl)
{
	ifstream file;
	file.open(fname,ifstream::in | ifstream::binary);
	if (file.is_open())
	{
		unsigned  int magic_number=0;
		unsigned  int number_of_labels=0;
		file.read((char*)&magic_number,sizeof(magic_number)); //если перевернуть то будет 2051
		endian_swap(magic_number);
		file.read((char*)&number_of_labels,sizeof(number_of_labels));//если перевернуть будет 10к
		endian_swap(number_of_labels);

		cout << "magic_number=" << magic_number << endl;
		cout << "number_of_labels=" << number_of_labels << endl;

		for(int i=0;i<number_of_labels;++i)
		{
			unsigned char t_ch=0;
			file.read((char*)&t_ch,sizeof(t_ch));
			vec_lbl.push_back(t_ch);
		}
	}
}
//-----------------------------------------------------------------------------------------------------
// Чтение изображений MNIST, на выходе вектор матриц с изображениями цифр
//-----------------------------------------------------------------------------------------------------
void read_mnist(string fname,vector<Mat>& vec_img)
{
	ifstream file;
	file.open(fname,ifstream::in | ifstream::binary);
	if (file.is_open())
	{
		unsigned  int magic_number=0;
		unsigned  int number_of_images=0;
		unsigned  int n_rows=0;
		unsigned  int n_cols=0;
		file.read((char*)&magic_number,sizeof(magic_number)); //если перевернуть то будет 2051
		endian_swap(magic_number);
		file.read((char*)&number_of_images,sizeof(number_of_images));//если перевернуть будет 10к
		endian_swap(number_of_images);
		file.read((char*)&n_rows,sizeof(n_rows));
		endian_swap(n_rows);
		file.read((char*)&n_cols,sizeof(n_cols));
		endian_swap(n_cols);
		cout << "Магическое число=" << magic_number << endl;
		cout << "Высота изображений=" << n_rows << endl;
		cout << "Ширина изображений=" << n_cols << endl;
		cout << "Количество изображений=" << number_of_images << endl;

		for(int i=0;i<number_of_images;++i)
		{
			Mat temp(n_rows,n_cols,CV_8UC1);
			for(int r=0;r<n_rows;++r)
			{
				for(int c=0;c<n_cols;++c)
				{
					unsigned char t_ch=0;
					file.read((char*)&t_ch,sizeof(t_ch));
					//тут идет запись матрицы 28х28 в вектор
					temp.at<unsigned char>(r,c)= t_ch; //получаем бинаризованные изображения
				}
			}
			vec_img.push_back(temp);
		}
	}

}

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
//-----------------------------------------------------------------------------------------------------
// Функция потерь и градиенты
// Векторизованный вариант
//-----------------------------------------------------------------------------------------------------
void sparseAutoencoderCost_v(Mat &W,int visibleSize,int hiddenSize,double lambda,double sparsityParam,double beta, Mat& data,double* cost,Mat& grad)
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
	Mat W1grad;
	Mat W2grad;
	Mat b1grad;
	Mat b2grad;

	double numPatches=data.cols;

	Mat avgActivations;

	double J=0;
	static Mat X,z2,z3,a2,a3,tmp;
	//----------------------------
	// прямой проход (расчет выхода сети)
	//----------------------------
	Mat B1(W1.rows,data.cols,CV_64FC1),B2(W2.rows,data.cols,CV_64FC1);
	for(int i=0;i<data.cols;i++)
	{
		b1.copyTo(B1.col(i));
		b2.copyTo(B2.col(i));
	}
	z2=(W1*data)+B1;
	a2=sigmoid(z2);
	z3=(W2*a2)+B2;
	a3=sigmoid(z3);
	pow( a3-data , 2 , tmp );
	J=0.5*sum(tmp)[0];

	cv::reduce(a2, avgActivations, 1, CV_REDUCE_AVG, CV_64FC1);
	avgActivations+=DBL_MIN;
	//----------------------------
	// Вычисления, связанные с условием разреженности
	// то есть чтобы в скрытом слое на поданный сигнал
	// активировалось лишь небольшое количество нейронов
	//----------------------------

	// Добавляется к дельте скрытого слоя при обратном проходе
	Mat sparsity_grad=beta*(-(sparsityParam/avgActivations)+((1.0-sparsityParam)/(1.0-avgActivations)));

	Mat S_G(sparsity_grad.rows,data.cols,CV_64FC1);
	for(int i=0;i<data.cols;i++)
	{
		sparsity_grad.copyTo(S_G.col(i));
	}
	// Слагаемые дивергенции Куллбэка-Лейблера

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
	Mat delta_3=(a3-data).mul(a3).mul(1-a3);
	//ошибка скрытого слоя
	Mat err2=W2.t()*delta_3+S_G;
	Mat delta_2=err2.mul(a2).mul(1-a2);
	//----------------------------
	// Градиенты весов
	//----------------------------
	W1grad=(delta_2*data.t())/numPatches+(lambda*W1);
	W2grad=(delta_3*a2.t())/numPatches+(lambda*W2);
	//----------------------------
	// Градиенты смещений
	//----------------------------
	cv::reduce(delta_2, b1grad, 1, CV_REDUCE_AVG, CV_64FC1);
	cv::reduce(delta_3, b2grad, 1, CV_REDUCE_AVG, CV_64FC1);
	//----------------------------
	// Соберем вычисленные значения
	// градиентов в вектор-столбец
	// (подходящий для minFunc).
	//----------------------------

	W1grad=W1grad.reshape(1,hiddenSize*visibleSize);
	W2grad=W2grad.reshape(1,hiddenSize*visibleSize);

	grad=Mat(W1grad.rows+W2grad.rows+b1grad.rows+b2grad.rows,1,CV_64FC1);
	W1grad.copyTo(grad(Range(0,hiddenSize*visibleSize),Range::all()));
	W2grad.copyTo(grad(Range(hiddenSize*visibleSize,2*hiddenSize*visibleSize),Range::all()));
	b1grad.copyTo(grad(Range(2*hiddenSize*visibleSize,2*hiddenSize*visibleSize+hiddenSize),Range::all()));
	b2grad.copyTo(grad(Range(2*hiddenSize*visibleSize+hiddenSize,grad.rows),Range::all()));

}
//-----------------------------------------------------------------------------------------------------
// Функция потерь и градиенты
// Реализация через циклы
//-----------------------------------------------------------------------------------------------------
void sparseAutoencoderCost_c(Mat &W,int visibleSize,int hiddenSize,double lambda,double sparsityParam,double beta, Mat& data,double* cost,Mat& grad)
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
	double  r = sqrt(6.0) / sqrt(((double)hiddenSize+(double)visibleSize+1.0));
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
void DrawNet(Mat& theta,Mat& patches_img,int hiddenSize,int visibleSize,int patch_side,int scale)
{
	// отобразим N_GridRows*N_GridCols первых патчей
	int N_GridRows=sqrtf(hiddenSize);
	int N_GridCols=N_GridRows;

	patches_img=Mat((patch_side+1)*N_GridRows*scale,(patch_side+1)*N_GridCols*scale,CV_32FC1); // Это для рисования поэтому 32FC1
	patches_img=0.5;

	Mat W1 = theta.rowRange(0,visibleSize*hiddenSize).reshape(1,hiddenSize); 
	W1=W1.t();
	//--------------------------------------------------
	//normalize(W1,W1,0,1,cv::NORM_MINMAX);
	for(int i=0;i<W1.cols;i++)
	{	
		Mat p_im;
		W1.col(i).copyTo(p_im);
		p_im=p_im.reshape(1,patch_side);
		normalize(p_im,p_im,0,1,cv::NORM_MINMAX);
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
//
//-----------------------------------------------------------------------------------------------------
class SparseAutoencoder
{
public:
//-----------------------------------------------------------------------------------------------------
// ПАРАМЕТРЫ СЕТИ
//-----------------------------------------------------------------------------------------------------
int NumPatches;
int patch_side;
int visibleSize;
int hiddenSize;
double lambda;
double sparsityParam;
double beta;
Mat theta; 
Mat Patches;
int scale; // степень увеличения
};
/*
//-----------------------------------------------------------------------------------------------------
// ПАРАМЕТРЫ СЕТИ
//-----------------------------------------------------------------------------------------------------
int NumPatches;
int patch_side;
int visibleSize;
int hiddenSize;
Mat theta;
double lambda;
double sparsityParam;
double beta;
*/
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void InitImages(SparseAutoencoder& AE)
{
AE.NumPatches=10000;
AE.patch_side=10;
AE.visibleSize=AE.patch_side*AE.patch_side;
AE.hiddenSize=25;
AE.lambda=1e-4;
AE.sparsityParam=0.01;
AE.beta=3;
AE.scale=5;
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void InitMNIST(SparseAutoencoder& AE)
{
AE.NumPatches=10000;
AE.patch_side=28;
AE.visibleSize=AE.patch_side*AE.patch_side;
AE.hiddenSize=169;
AE.lambda=3e-3;
AE.sparsityParam=0.1;
AE.beta=3;
AE.scale=1;
}

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
	SparseAutoencoder* AE = (SparseAutoencoder*)instance;

	Mat grad;// градиент
	lbfgsfloatval_t fx = 0.0; // Целевая (минимизируемая) функция

	//sparseAutoencoderCost_c(theta,visibleSize,hiddenSize,lambda,sparsityParam,beta, Patches,&fx,grad);
	sparseAutoencoderCost_v(AE->theta,AE->visibleSize,AE->hiddenSize,AE->lambda,AE->sparsityParam,AE->beta, AE->Patches,&fx,grad);
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
	SparseAutoencoder* AE = (SparseAutoencoder*)instance;
	// *x - указывает равен theta.data, поэтому копирования не требуется
	// Рисуем его
	Mat patches_img;
	DrawNet(AE->theta,patches_img,AE->hiddenSize,AE->visibleSize,AE->patch_side,AE->scale);
	patches_img.convertTo(patches_img,CV_32FC1);
	imshow("Patches",patches_img);
	waitKey(15);

	printf("Iteration %d:\n", k);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}

//-----------------------------------------------------------------------------------------------------
// Инициализация параметров
//-----------------------------------------------------------------------------------------------------
void InitParameters(int N,vector<Mat>& MNIST,Mat& Features)
{
	int h=MNIST[0].cols;
	int w=MNIST[0].rows;

	Features=Mat(h*w,N,CV_64FC1,1.0/255.0);
	Mat tmp;
	// Это делается чтобы можно было задать объем обрабатываемых данных (через N)
	for(int i=0;i<N;i++)
	{		
		tmp=MNIST[i].reshape(1,h*w);
		tmp.convertTo(Features.col(i),CV_64FC1);
	}
}
//-----------------------------------------------------------------------------------------------------
// Загрузка патчей из датасета MNIST
//-----------------------------------------------------------------------------------------------------
void SetPatchesFromMNIST(SparseAutoencoder& AE)
{
	setlocale(LC_ALL, "Russian");
	vector<Mat> MNIST;
	// читаем изображения
	cout << "Загрузка изображений." << endl;
	read_mnist("D:/MNIST/t10k-images.idx3-ubyte",MNIST);
	cout << "Загрузка изображений выполнена." << endl;
	// создаем окно
	namedWindow("result");
	// размеры одного изображения
	int rows=MNIST[0].rows;
	int cols=MNIST[0].cols;

	// Создаем и заполняем необходимые переменные
	cout << "Инициализация параметров." << endl;
	InitParameters(AE.NumPatches,MNIST,AE.Patches);
	cout << "Инициализация параметров выполнена." << endl;
}
//-----------------------------------------------------------------------------------------------------
// Нарезка патчей из 10 изображений, предварительно обработанных whitening 
//-----------------------------------------------------------------------------------------------------
void SetPatchesFromImages(SparseAutoencoder& AE)
{
	// Здесь будут изображения с которых будем брать патчи
	vector<Mat> Images;
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
	double n_patches=AE.NumPatches;
	AE.Patches=Mat(AE.patch_side*AE.patch_side,n_patches,CV_64FC1);
	RNG rng;
	for(int i=0;i<n_patches;i++)
	{
		int n_img=rng.uniform(0.0,1.0)*((double)Images.size()-1);
		int row=rng.uniform(0.0,1.0)*((double)Images[n_img].rows-AE.patch_side);
		int col=rng.uniform(0.0,1.0)*((double)Images[n_img].cols-AE.patch_side);
		Mat patch=Images[n_img](Range(row,row+AE.patch_side),Range(col,col+AE.patch_side)).clone();
		patch=patch.reshape(1,AE.patch_side*AE.patch_side).clone();
		patch.copyTo(AE.Patches.col(i));
	}
}
//-----------------------------------------------------------------------------------------------------
// Нормализация патчей
//-----------------------------------------------------------------------------------------------------
void NormalizePatches(Mat& Patches)
{
	// -------------------------------------------------------------------------------
	// Нормализация 
	// (выравниваем по центру,
	// отсекаем выбросы,
	// приводим к интервалу 0.1-0.9, т.к на выходе у нас сигмоидные функции активации)
	// -------------------------------------------------------------------------------
	for(int j=0;j<Patches.cols;j++)
	{
	Scalar mean;
	Scalar stddev;
	cv::meanStdDev(Patches.col(j),mean,stddev);
		Patches.col(j)-=mean[0];

		for(int i=0;i<Patches.rows;i++)
		{
			if(Patches.at<double>(i,j)>3.0*stddev[0]){Patches.at<double>(i,j)=3.0*stddev[0];}
			if(Patches.at<double>(i,j)<-3.0*stddev[0]){Patches.at<double>(i,j)=3.0*stddev[0];}
		}

		if(stddev[0]!=0)
		{
		Patches.col(j)/=(3.0*stddev[0]);
		}
	}
	Scalar Mean;
	Scalar Stddev;
	cv::meanStdDev(Patches,Mean,Stddev);
	Patches-=Mean;
	Patches/=3*Stddev[0];
	//Patches=(Patches+1)*0.4+0.1;
	normalize(Patches,Patches,0.1,0.9,cv::NORM_MINMAX); // приведем к удобному диапазону
}
//-----------------------------------------------------------------------------------------------------
// Точка входа
//-----------------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	SparseAutoencoder AE;
	//Mat Patches;
	// ----------------------------------------------
	// Инициализация генератора случайных чисел
	// ----------------------------------------------
	srand((unsigned)time(NULL));
	
	// Раскомментировать, если нарезка из изображений

	//InitImages(AE);
	//SetPatchesFromImages(AE);
	
	// Раскомментировать, если загрузка из MNIST

	InitMNIST(AE);
	SetPatchesFromMNIST(AE);
	
	NormalizePatches(AE.Patches);
	// -------------------------------------------------------------------------------
	// Инициализация вектора параметров небольшими случайными величинами
	// -------------------------------------------------------------------------------
	AE.theta=initializeParameters(AE.hiddenSize,AE.visibleSize);

	int ret = 0;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x = lbfgs_malloc(AE.theta.rows);
	lbfgs_parameter_t param;
	// Инициализация параметров L-BFGS оптимизатора. 
	lbfgs_parameter_init(&param);
	// Предел точности
	param.epsilon=1e-11;
	// Количество итераций
	param.max_iterations=200000;
	// Метод линейного поиска 
	//param.linesearch=LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
	param.linesearch=::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
	
	//
	// Начало L-BFGS оптимизации; используются функции обратного вызова
	// evaluate() для вычислния минимизируемого функционала и градиентов
	// и progress(), для отображения промежуточных результатов оптимизации.
	//

	ret = lbfgs(AE.theta.rows,(lbfgsfloatval_t *) AE.theta.data, &fx, evaluate, progress, &AE, &param);

	// Report the result.
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);

	lbfgs_free(x);

	waitKey(0);

	return 0;
}