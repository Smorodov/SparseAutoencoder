function result=whiten(image, f0, n);
% function RESULT=WHITEN(IMAGE, F0, N);
% 
% Performs image whitening as described in Olshausen & Field's
% Vision Research article (1997)
%
% F0 and N are optional parameters - if not, Olshausen &
% Field's values are used. F0 controls the radius of the decay
% function. N controls the steepness of the radial decay. 
%
% 'whiten.m'
% Copyright 2005, 
% Nicholas Butko, Graduate Student, UCSD Dept. of Cognitive Science
if nargin == 1
    f0=.4*min(size(image));
    n=4;
elseif nargin ==2
    n=4;
end

size1=min(size(image)); 
size2=min(size(image));

start1=ceil(-(size1-1)/2);
start2=ceil(-(size2-1)/2);
end1=ceil((size1-1)/2);
end2=ceil((size2-1)/2);

[kx ky]=meshgrid(start1:end1,start2:end2);
r=sqrt(kx.^2+ky.^2).*exp(-(sqrt(kx.^2+ky.^2)./f0).^n);
im=fftshift(fft2(image(1:size2,1:size1)));
con=im.*r;
result=real(ifft2(ifftshift(con)));