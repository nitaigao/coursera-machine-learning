function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
   %h = X * theta;
   %g = (h - y) .* X(:, 2);
   %theta = theta - ((alpha / m) * g);
   
   % theta0 + theta1 * x
   h = X * theta;
   
   %h - y
   xmtsy = h - y;
   
   % (alpha / m) is the same as alpha * 1/m;
   pd = (alpha / m);
   
   % sums all the iterations of of (h - y) * x
   summed_results = (X' * xmtsy);
   
   % accumulate the final result
   theta = theta - pd * summed_results;
        
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
