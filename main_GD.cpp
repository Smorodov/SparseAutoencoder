#include <iostream>
#include <vector>
#include <stdio.h>
#include <Eigen/Dense>
#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/eigen.hpp"
#include "fstream"
#include "iostream"
#include <time.h>
#include <stdlib.h>
using namespace std;
using namespace cv;
using namespace Eigen;
const double E=2.7182818284590452353602874713527;

//-----------------------------------------------------------------------------------------------------
// Сигмоид (поэлементный) матрицы
//-----------------------------------------------------------------------------------------------------
Mat sigmoid(Mat& x)
{
	Mat res(x.rows,x.cols,CV_32FC1);

	int nRows = x.rows;
	int nCols = x.cols;
	int i,j;
	float* px;
	float* py;
	for( i = 0; i < nRows; ++i)
	{
		px = x.ptr<float>(i);
		py = res.ptr<float>(i);
		for ( j = 0; j < nCols; ++j)
		{
			py[j] = 1.0/(1+powl(E,-px[j]));
		}
	}
	//cout << res << endl;
	return res;
}
//-----------------------------------------------------------------------------------------------------
// Округление
//-----------------------------------------------------------------------------------------------------
int round (double value)
{
	return static_cast<int>(floor(value + 0.5));
}
//-----------------------------------------------------------------------------------------------------
// Берем число из интервала [0,1) согласно равномерному распределению
//-----------------------------------------------------------------------------------------------------
double Rand(void)
{
	return double(rand())/double(RAND_MAX);
}

//----------------------------------------------------------
// Функция для перестановки четвертей изображения местамии
// так, чтобы ноль спектра находился в центре изображения.
//----------------------------------------------------------
void Recomb(Mat &src,Mat &dst)
{
	int cx = src.cols>>1;
	int cy = src.rows>>1;
	Mat tmp;
	tmp.create(src.size(),src.type());
	src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
	src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));	
	src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
	src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
	dst=tmp;
}
//----------------------------------------------------------
// По заданному изображению рассчитывает 
// действительную и мнимую части спектра Фурье
//----------------------------------------------------------
void ForwardFFT(Mat &Src, Mat *FImg)
{
	int M = getOptimalDFTSize( Src.rows );
	int N = getOptimalDFTSize( Src.cols );
	Mat padded;    
	copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
	// Создаем комплексное представление изображения
	// planes[0] содержит само изображение, planes[1] его мнимую часть (заполнено нулями)
	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg); 
	dft(complexImg, complexImg);	
	// После преобразования результат так-же состоит из действительной и мнимой части
	split(complexImg, planes);

	// обрежем спектр, если у него нечетное количество строк или столбцов
	planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
	planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));

	Recomb(planes[0],planes[0]);
	Recomb(planes[1],planes[1]);
	// Нормализуем спектр
	planes[0]/=float(M*N);
	planes[1]/=float(M*N);
	FImg[0]=planes[0].clone();
	FImg[1]=planes[1].clone();
}
//----------------------------------------------------------
// По заданным действительной и мнимой части
// спектра Фурье восстанавливает изображение
//----------------------------------------------------------
void InverseFFT(Mat *FImg,Mat &Dst)
{
	Recomb(FImg[0],FImg[0]);
	Recomb(FImg[1],FImg[1]);
	Mat complexImg;
	merge(FImg, 2, complexImg);
	// Производим обратное преобразование Фурье
	idft(complexImg, complexImg);
	split(complexImg, FImg);
	normalize(FImg[0], Dst, 0, 1, CV_MINMAX);
}
//----------------------------------------------------------
// Раскладывает изображение на амплитуду и фазу спектра Фурье
//----------------------------------------------------------
void ForwardFFT_Mag_Phase(Mat &src, Mat &Mag,Mat &Phase)
{
	Mat planes[2];
	ForwardFFT(src,planes);
	Mag.zeros(planes[0].rows,planes[0].cols,CV_32F);
	Phase.zeros(planes[0].rows,planes[0].cols,CV_32F);
	cv::cartToPolar(planes[0],planes[1],Mag,Phase);
}
//----------------------------------------------------------
// По заданным амплитуде и фазе
// спектра Фурье восстанавливает изображение
//----------------------------------------------------------
void InverseFFT_Mag_Phase(Mat &Mag,Mat &Phase, Mat &dst)
{
	Mat planes[2];
	planes[0].create(Mag.rows,Mag.cols,CV_32F);
	planes[1].create(Mag.rows,Mag.cols,CV_32F);
	cv::polarToCart(Mag,Phase,planes[0],planes[1]);
	InverseFFT(planes,dst);
}
//----------------------------------------------------------
// Whiten image
//----------------------------------------------------------
void whiten(Mat& src,Mat &dst)
{
	double f0=200;
	//----------------------------------------------------------
	// Производим свертку
	//----------------------------------------------------------
	// Раскладываем изображение в спектр
	// Ампльтуда спектра
	Mat Mag;
	// Фаза спектра
	Mat Phase;
	ForwardFFT_Mag_Phase(src,Mag,Phase);
	//----------------------------------------------------------
	// Создаем частотный фильтр
	//----------------------------------------------------------
	Mat filter;
	filter=Mat::zeros(Mag.rows,Mag.cols,CV_32F);
	for(int i=0;i<src.rows;i++)
	{
		double I=(float)i-((float)src.rows-1)/2.0;
		float* F = filter.ptr<float>(i);
		for(int j=0;j<src.cols;j++)
		{
			double J=(float)j-((float)src.cols-1)/2.0;
			double f=sqrtl(I*I+J*J);
			F[j]=f*powf(E,-powf((f/f0),4.0));
		}
	}


	cv::multiply(Mag,filter,Mag);
	//cv::multiply(Phase,filter,Phase);
	//----------------------------------------------------------
	// Обратное преобразование
	//----------------------------------------------------------
	InverseFFT_Mag_Phase(Mag,Phase,dst);
	dst=dst(Range(0,src.rows),Range(0,src.cols));
}


void sparseAutoencoderCost(Mat &W,int visibleSize,int hiddenSize,double lambda,double sparsityParam,double beta, vector<Mat>& data,double& cost,Mat& grad)
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
	Mat W1grad = Mat::zeros(W1.rows,W1.cols,CV_32FC1);
	Mat W2grad = Mat::zeros(W2.rows,W2.cols,CV_32FC1);
	Mat b1grad =  Mat::zeros(b1.rows,b1.cols,CV_32FC1);
	Mat b2grad =  Mat::zeros(b2.rows,b2.cols,CV_32FC1);

	// ---------- ВАШ КОД ЗДЕСЬ --------------------------------------
	//  Инструкции: Вычислите функцию потерь/функцию оптимизации J_sparse(W,b) для разреженного автонкодера,
	//              и соотретствующие градиенты W1grad, W2grad, b1grad, b2grad.
	//
	// W1grad, W2grad, b1grad и b2grad вычисляются методом обратного распространения ошибки.
	// Заметте, что W1grad должна иметь те же размеры что и W1,
	// b1grad должна иметь те же размеры что и b1, и т.д.
	// W1grad это частная производная J_sparse(W,b) по W1.
	// Т.е., W1grad(i,j) это частная производная J_sparse(W,b)
	// по параметру W1(i,j).  Таким образом, W1grad должна быть равна
	// [(1/m) Delta W1 + lambda W1] в последнем блоке псевдокода секция 2.2
	// лекций (и аналогично для W2grad, b1grad, b2grad).
	//
	// Другими словами, если мы используем пакетный метод градиентного спуска,
	// на каждом шаге W1 будет уточняться по формуле: W1 := W1 - alpha * W1grad,
	// аналогично для W2, b1, b2.
	//
	// i - номер фрагмента (образца данных)

	double numPatches=data.size();

	Mat avgActivations=Mat::zeros(W1.rows,1,CV_32FC1);
	Mat storedHiddenValues = Mat::zeros(hiddenSize, numPatches,CV_32FC1);
	Mat storedOutputValues = Mat::zeros(visibleSize, numPatches,CV_32FC1);
	double J=0;
	Mat X,z2,z3,a2,a3;
	//----------------------------
	// прямой проход (расчет выхода сети)
	//----------------------------
	for (int i=0;i<numPatches;i++)
	{
		data[i].copyTo(X);
		z2=W1*X+b1;
		a2=sigmoid(z2);
		avgActivations=avgActivations+a2;
		z3=W2*a2+b2;
		a3=sigmoid(z3);
		// сохраним результаты прямого хода
		a2.copyTo(storedHiddenValues.col(i));
		a3.copyTo(storedOutputValues.col(i));
		// Слагаемое функции потерь (сумма квадратов ошибки)
		Mat tmp;
		pow(a3-X,2,tmp);
		//tmp=tmp.mul(tmp);
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
	Mat sparsity_grad=beta*(-sparsityParam/avgActivations+((1-sparsityParam)/(1-avgActivations)));

	// Слагаемые дивергенции Куллбэка-Лейблера
	Mat tmp1;
	log(sparsityParam/avgActivations,tmp1);
	Mat KL1=sparsityParam*tmp1;
	log((1-sparsityParam)/(1-avgActivations),tmp1);
	Mat KL2=(1-sparsityParam)*tmp1;
	// дивергенция Куллбэка-Лейблера (сумма элементов по всей выборке данных)
	double KL_divergence=sum(KL1+KL2)[0];
	// Функция потерь (минимизируемый функционал)

	Mat tmp3;
	cv::pow(W1,2,tmp3);
	Mat tmp4;
	cv::pow(W2,2,tmp4);
	cost=(1/numPatches)*J+lambda*0.5*(sum(tmp3)[0]+sum(tmp4)[0])+beta*KL_divergence;
	//----------------------------
	// обратное распространение ошибки
	//----------------------------

	for (int i=0;i<numPatches;i++)
	{
		data[i].copyTo(X);
		// достаем ранее сохраненные веса
		//Mat a2;
		storedHiddenValues.col(i).copyTo(a2);
		//Mat a3;
		storedOutputValues.col(i).copyTo(a3);

		// ошибка выходного слоя
		Mat delta_3=(a3-X).mul(a3.mul(1-a3));
		// ошибка скрытого слоя
		Mat delta_2=(W2.t()*delta_3+sparsity_grad).mul(a2.mul(1-a2));
		//cout << a2 << endl;
		W1grad+=delta_2*X.t();
		W2grad+=delta_3*a2.t();

		b1grad+=delta_2;
		b2grad+=delta_3;
	}

	//----------------------------
	// Градиенты весов
	//----------------------------
	W1grad=(1.0/numPatches)*W1grad+(lambda)*W1;
	W2grad=(1.0/numPatches)*W2grad+(lambda)*W2;
	//----------------------------
	// Градиенты смещений
	//----------------------------
	b1grad = (1.0/numPatches)*b1grad;
	b2grad = (1.0/numPatches)*b2grad;
	//----------------------------
	// Соберем вычисленные значения
	// градиентов в вектор-столбец
	// (подходящий для minFunc).
	//----------------------------

	W1grad=W1grad.reshape(1,hiddenSize*visibleSize);
	W2grad=W2grad.reshape(1,hiddenSize*visibleSize);
	b1grad=b1grad.reshape(1,hiddenSize);
	b2grad=b2grad.reshape(1,visibleSize);

	grad=Mat(W1grad.rows+W2grad.rows+b1grad.rows+b2grad.rows,1,CV_32FC1);
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
	double  r = sqrt(6.0) / (sqrt((double)hiddenSize+(double)visibleSize+1.0));
	// делаем выборку из интервала [-r, r] по равномерному закону распределения
	//W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
	Mat W1=Mat(hiddenSize,visibleSize,CV_32FC1);
	cv::randu(W1,-r,r);
	Mat W2=Mat(visibleSize,hiddenSize,CV_32FC1);
	cv::randu(W2,-r,r);
	//W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
	// смещения заполним нулями
	Mat b1 = Mat::zeros(hiddenSize, 1,CV_32FC1);
	Mat b2 = Mat::zeros(visibleSize, 1,CV_32FC1);
	// Переведем веса и смещения в форму вектора-столбца.
	// Этот шаг "разворачивает" все параметры (веса и смещения) в один вектор  
	// который может использоваться функцией minFunc. 

	//theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
	W1=W1.reshape(1,hiddenSize*visibleSize);
	W2=W2.reshape(1,hiddenSize*visibleSize);
	b1=b1.reshape(1,hiddenSize);
	b2=b2.reshape(1,visibleSize);
	theta=Mat(W1.rows+W2.rows+b1.rows+b2.rows,1,CV_32FC1);
	W1.copyTo(theta(Range(0,hiddenSize*visibleSize),Range::all()));
	W2.copyTo(theta(Range(hiddenSize*visibleSize,2*hiddenSize*visibleSize),Range::all()));
	b1.copyTo(theta(Range(2*hiddenSize*visibleSize,2*hiddenSize*visibleSize+hiddenSize),Range::all()));
	b2.copyTo(theta(Range(2*hiddenSize*visibleSize+hiddenSize,theta.rows),Range::all()));

	return theta;
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void DrawNet(Mat& net,Mat& patches_img,int hiddenSize,int visibleSize,int patch_side,int scale)
{
	// отобразим N_GridRows*N_GridCols первых патчей
	int N_GridRows=sqrtf(hiddenSize);
	int N_GridCols=N_GridRows;

	patches_img=Mat((patch_side+1)*N_GridRows*scale,(patch_side+1)*N_GridCols*scale,CV_32FC1);
	patches_img=0;
	Mat p=(net(Range(0,visibleSize*hiddenSize),Range::all()));

	p=p.reshape(1,hiddenSize).clone();

	for(int i=0;i<N_GridRows*N_GridCols;i++)
	{	
		Mat p_im;
		p.row(i).copyTo(p_im);
		p_im=p_im.reshape(1,patch_side);
		normalize(p_im,p_im,0.1,0.9,cv::NORM_MINMAX); // приведем к удобному диапазону
		//p_im.copyTo(patches_img(Range((i/N_GridCols)*(patch_side+1),(i/N_GridCols)*(patch_side+1)+patch_side),Range((i%N_GridCols)*(patch_side+1),(i%N_GridCols)*(patch_side+1)+patch_side)));

		for(int j=0;j<p_im.rows;j++)
		{
			for(int k=0;k<p_im.cols;k++)
			{
				Point pt1=Point((i/N_GridCols)*(patch_side+1)*scale+k*scale+0.5*scale,(i%N_GridCols)*(patch_side+1)*scale+j*scale+0.5*scale);
				Point pt2=Point(((i/N_GridCols)*(patch_side+1)+1)*scale+k*scale+0.5*scale,((i%N_GridCols)*(patch_side+1)+1)*scale+j*scale+0.5*scale);
				rectangle(patches_img,pt1,pt2,Scalar::all(p_im.at<float>(j,k)),-1);
			}
		}



	}
}

//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
int main( int argc, char** argv )
{
	srand((unsigned)time(NULL));
	double patch_side=8;
	double visibleSize = patch_side*patch_side;		// Количество входных узлов 
	double hiddenSize = 25;							// количество скрытых узлов 
	double sparsityParam = 0.01;				// желаемый средний уровень активации нейронов скрытого слоя.
	// (этот параметр обозначен в лекции греческой буквой ро). 
	double lambda = 0.0001;						// коэффициент ослабления весов связей       
	double beta = 3;							// коэффициент разреженности   

	Mat patches_img;
	cv::Mat_<float> a = Mat_<float>::ones(2,2);
	Eigen::Matrix<float,Dynamic,Dynamic> b;
	cv2eigen(a,b);
	eigen2cv(b,a);

	// Здесь будут фильтры
	Mat Filters;
	// Здесь будут изображения с которых будем брать патчи
	vector<Mat> Images;
	// Это сами патчи
	vector<Mat> Patches;
	namedWindow("Image");
	namedWindow("Patches");
	// Считаем изображения, с которых будем брать патчи

	for(int i=0;i<10;i++)
	{
		stringstream str;
		str << "..\\images\\image"<<i<<".jpg";
		Mat img;
		imread(str.str(),0).convertTo(img,CV_32FC1,1.0/255.0);
		Images.push_back(img);
		cout << str.str() << endl;
	}

	// теперь набираем патчи
	double n_patches=1000;

	for(int i=0;i<n_patches;i++)
	{
		int n_img=Rand()*((double)Images.size()-1);
		int row=Rand()*((double)Images[n_img].rows-patch_side);
		int col=Rand()*((double)Images[n_img].cols-patch_side);
		Mat patch=Images[n_img](Range(row,row+patch_side),Range(col,col+patch_side)).clone();
		patch=patch.reshape(1,patch_side*patch_side).clone();
		normalize(patch,patch,0.2,0.8,cv::NORM_MINMAX); // приведем к удобному диапазону
		Patches.push_back(patch);
	}

	// Инициализация вектора параметров небольшими случайными величинами
	Mat theta=initializeParameters(hiddenSize,visibleSize);
	Mat grad;
	double cost;

	for(int i=0;i<40000;i++)
	{
		sparseAutoencoderCost(theta,visibleSize,hiddenSize,lambda,sparsityParam,beta, Patches,cost,grad);
		theta-=grad;
		DrawNet(theta,patches_img,hiddenSize,visibleSize,patch_side,8);
		//cv::resize(patches_img,patches_img,Size(patches_img.cols*2,patches_img.rows*2),1);
		imshow("Patches",patches_img);
		waitKey(15);

		cout << cost << endl;
	}


	waitKey(0);

	return 0;
}