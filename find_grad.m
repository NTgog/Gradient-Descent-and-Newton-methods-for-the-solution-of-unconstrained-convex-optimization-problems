function grad = find_grad(c, b, A, x)

 % Gradient of function c.'*x-sum(log(b-A*x))

     m = size(b,1);

     grad = c;

     for i = (1:1:m)
         a_i = A(i,:).';
         b_i = b(i);
         grad = grad + (1/(b_i-a_i.'*x)).*a_i;
     end

end

