function [cost,grad] = sparseAutoencoderCost(W, visibleSize, hiddenSize, ...
    lambda, sparsityParam, beta, data)

% visibleSize: количество входных узлов (probably 64)
% hiddenSize: количество скрытых узнов (probably 25)
% lambda: коэффициент ослабления весов
% sparsityParam: Желаемый уровень активации нейронов скрытого слоя (Ро).
% beta: Коэффициент (вес) слагаемого отвечающего за разреженность.
% data: Наша матрица 64x10000 содержащая обучающую выборку.
% таким образом, data(:,i) это i-th обучающая пара (вход и выход, в данном случае одно и то-же).

% Входной параметр W это вектор (т.к. minFunc ожидает, что параметр является вектором).
% Сначала мы разобъем W на куски (W1, W2, b1, b2), чтобы все было как в лекции.

% Веса, соединяющие входной вектор и скрытый слой
W1 = reshape(W(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% Веса, соединяющие скрытый слой и выход
W2 = reshape(W(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% Смещения нейронов скрытого слоя
b1 = W(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% Смещения нейронов выходного слоя
b2 = W(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Функция стоимости и градиенты (Ваш код должен рассчитать эти значения).
% Тут все инициализируется нулями.
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

% ---------- ВАШ КОД ЗДЕСЬ --------------------------------------
%  Инструкции: Вычислите функцию потерь/функцию оптимизации J_sparse(W,b) для разреженного автонкодера,
%              и соотретствующие градиенты W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad и b2grad вычисляются методом обратного распространения ошибки.
% Заметте, что W1grad должна иметь те же размеры что и W1,
% b1grad должна иметь те же размеры что и b1, и т.д.
% W1grad это частная производная J_sparse(W,b) по W1.
% Т.е., W1grad(i,j) это частная производная J_sparse(W,b)
% по параметру W1(i,j).  Таким образом, W1grad должна быть равна
% [(1/m) Delta W1 + lambda W1] в последнем блоке псевдокода секция 2.2
% лекций (и аналогично для W2grad, b1grad, b2grad).
%
% Другими словами, если мы используем пакетный метод градиентного спуска,
% на каждом шаге W1 будет уточняться по формуле: W1 := W1 - alpha * W1grad,
% аналогично для W2, b1, b2.
%
% i - номер фрагмента (образца данных)

numPatches=size(data,2);

avgActivations=zeros(size(W1,1),1);
storedHiddenValues = zeros(hiddenSize, numPatches);
storedOutputValues = zeros(visibleSize, numPatches);
J=0;
%----------------------------
% прямой проход (расчет выхода сети)
%----------------------------
for i=1:numPatches
    X=data(:,i);
    z2=W1*X+b1;
    a2=sigmoid(z2);
    avgActivations=avgActivations+a2;
    z3=W2*a2+b2;
    a3=sigmoid(z3);
    % сохраним результаты прямого хода
    storedHiddenValues(:, i) = a2;
    storedOutputValues(:, i) = a3;
    % Слагаемое функции потерь (сумма квадратов ошибки)
    J=J+0.5*sum(sum((a3-X).^2));
end
%----------------------------
% Вычисления, связанные с условием разреженности
% то есть чтобы в скрытом слое на поданный сигнал
% активировалось лишь небольшое количество нейронов
%----------------------------

% из известной суммы найдем среднее
avgActivations=avgActivations./numPatches;
% Добавляется к дельте скрытого слоя при обратном проходе
sparsity_grad=beta.*(-sparsityParam./avgActivations+((1-sparsityParam)./(1-avgActivations)));

% Слагаемые дивергенции Куллбэка-Лейблера
KL1=sparsityParam*log(sparsityParam./avgActivations);
KL2=(1-sparsityParam)*log((1-sparsityParam)./(1-avgActivations));
% дивергенция Куллбэка-Лейблера (сумма элементов по всей выборке данных)
KL_divergence=sum(sum(KL1+KL2));
% Функция потерь (минимизируемый функционал)
cost=(1/numPatches)*J+lambda*0.5*(sum(sum(W1.^2))+sum(sum(W2.^2)))+beta*KL_divergence;
%----------------------------
% обратное распространение ошибки
%----------------------------
for i=1:numPatches
    X=data(:,i);
    % достаем ранее сохраненные веса
    a2 = storedHiddenValues(:, i);
    a3 = storedOutputValues(:, i);
    % ошибка выходного слоя
    delta_3=(a3-X).*a3.*(1-a3);
    % ошибка скрытого слоя
    delta_2=(W2'*delta_3+sparsity_grad).*a2.*(1-a2);
    
    W1grad=W1grad+delta_2*X';
    W2grad=W2grad+delta_3*a2';
    
    b1grad=b1grad+delta_2;
    b2grad=b2grad+delta_3;
end

%----------------------------
% Градиенты весов
%----------------------------
W1grad=(1/numPatches).*W1grad+(lambda).*W1;
W2grad=(1/numPatches).*W2grad+(lambda).*W2;
%----------------------------
% Градиенты смещений
%----------------------------
b1grad = (1/numPatches).*b1grad;
b2grad = (1/numPatches).*b2grad;
%----------------------------
% Соберем вычисленные значения
% градиентов в вектор-столбец
% (подходящий для minFunc).
%----------------------------
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Функция вычисления сигмоида
%-------------------------------------------------------------------
function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end