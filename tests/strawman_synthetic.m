function F = strawman_synthetic(x,Set,alpha)

% F is 1 x m

%Set defines the component functions of F you would like 
if nargin<2 % Use all components
	Set = 1:length(x); %(in this case there are dim n components)
elseif length(Set) > length(unique(Set)) 
	disp('Warning: Set has nonunique entries')
	return
end

x=x(:); % Turn into column vector

n = length(x); m = n;

F = zeros(1, m);
for j = 1:m
    F(j) = 2^j * (x(j) - j)^2; %(x(j) - 2^j)^2;
end

% if using sos code:
F = F(Set);
end