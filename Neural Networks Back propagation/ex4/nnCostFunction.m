function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can +verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Finding the hypothesis using zero thetas lol- feed forward propagation!!

a1=zeros(hidden_layer_size,m);
X=[ones(m,1) X];
a1=sigmoid(Theta1*X');
a1=a1';

a1=[ones(m,1) a1];
h=sigmoid(Theta2*a1');
a1size=size(a1)
% making the y matrix
y_new=zeros(num_labels,m);
for i=1:m
    for j=1:m
if y(i) ==j
    y_new(j,i)=1;
end
    end
end
y=y_new;
sum1=1;
% getting cost function using h 
for i=1:m
    J1(i)=log(h(:,i))'*y(:,i)+log(1-h(:,i))'*(1-y(:,i));
end
J=-1/m*sum(J1,2);

% adding regularization 

Theta1_sq=Theta1.^2;
Theta1_sq(:,1)=0;
Theta2_sq=Theta2.^2;
Theta2_sq(:,1)=0;
s1=sum(sum(Theta1_sq,1),2)+sum(sum(Theta2_sq,1),2);
J=J+lambda*s1/(2*m);

% finding gradient using back-propagation algorithm

del_3=h-y;
d3=size(del_3)

t1=size(Theta1)
t2=size(Theta2)

z2=Theta1*X';
sz2=size(z2)

t2=size(Theta2)
del_2=(Theta2(:,2:end)'*del_3).*sigmoidGradient(z2);
d2=size(del_2)

B_delta1=del_2*X;
B_delta2=del_3*a1;

bd1=size(B_delta1)
bd2=size(B_delta2)
del2=size(del_2)
aone=size(a1)


Theta1(:,1)=0;
Theta1_grad=B_delta1/m+lambda/m*Theta1;
Theta2(:,1)=0;
Theta2_grad=B_delta2/m+lambda/m*Theta2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
