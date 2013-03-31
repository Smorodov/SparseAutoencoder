function [] = checkNumericalGradient()
% Эта функция используется для проверки Вашей реализации функции вычисления
% градиента численным методом.
% В расположенной файле computeNumericalGradient.m
% Она аналитически вычисляет градиент очень простой функции, называемой
% simpleQuadraticFunction (см. ниже) и сравнивает результат с Вашим
% численным решением. Вашей реализации функции вычисления
% градиента неверна если решения (аналитическое и численное, реализованное Вами)
% расходятся на значительную величину.
 
% Вычисляем функцию и градиент в точке x = [4; 10];
x = [4; 10];
[value, grad] = simpleQuadraticFunction(x);

% Используем Ваш код для численного вычисления градиента функции 
% simpleQuadraticFunction в точке x.
% (Обозначение "@simpleQuadraticFunction" является указателем на функцию.)
numgrad = computeNumericalGradient(@simpleQuadraticFunction, x);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be 2.1452e-12 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');
end

function [value,grad] = simpleQuadraticFunction(x)
% this function accepts a 2D vector as input. 
% Its outputs are:
%   value: h(x1, x2) = x1^2 + 3*x1*x2
%   grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2 
% Note that when we pass @simpleQuadraticFunction(x) to computeNumericalGradients, we're assuming
% that computeNumericalGradients will use only the first returned value of this function.
value = x(1)^2 + 3*x(1)*x(2);
grad = zeros(2, 1);
grad(1)  = 2*x(1) + 3*x(2);
grad(2)  = 3*x(1);
end
