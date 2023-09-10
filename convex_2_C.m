%-------------------------------------------------------------------------%
%
%                        Convex Optimization 
%
%                           PROJECT 2
%
%                   Toganidis Nikos - 2018030085
%                Tzimpimpaki Evaggelia - 2018030165
%
%-------------------------------------------------------------------------%

clear all; close all; clc; %#ok<CLALL>



%% EXERCISE C.a

  n = 2;
  m = 20;

  A = randn(m,n); 
  c = rand(n,1);
  b = rand(m,1);

  % Create function f(x) = c^T *x - sum(log(b-Ax))
  f = @(x) transpose(c)*x - sum(log(b-A*x)); 

  % Gradient of f
  gradient_of_f = @(x) find_grad(c,b,A,x); 
  
  
  % Minimize f using CVX
    
    cvx_begin quiet 
        variable x(n) 
        minimize( f(x) ) 
    cvx_end 
    
    echo off
    
    x_star = x;
    p_star= f(x_star);

    fprintf(' \n');  
    fprintf(' <strong> C.a   </strong> Optimal value (CVX) = %4f \n',p_star);
    fprintf(' \n');  
    
    
%% EXERCISE C.b
    
    %  Make sure that n = 2. Contour plot f
    
    if (n == 2)
                
        [x1,x2] = meshgrid(x_star(1)-0.9:0.01:x_star(1)+0.9,x_star(2)-0.9:0.01:x_star(2)+0.9);

        f_vals = zeros(size(x1,1),size(x1,2));

        for i = 1 : size(x1,1)
            
            for j = 1 : size(x2,2)
                
                X = [x1(i,j) ; x2(i,j)];
                
                if (b-A*X > 0)
                    f_vals(i,j) = f(X);
                else 
                    f_vals(i,j) = 10^3;
                end
                
            end
            
        end

        figure(); mesh(x1,x2,f_vals);
        xlabel('{\bf x1}');
        ylabel('{\bf x2}');
        zlabel('{\bf f(x)}');
        titleText = 'Plot $f(x) = c^Tx - \sum{log(b-Ax)}$';
        title(titleText,'Interpreter','latex');
    
        figure();
        contour(x1,x2,f_vals);
        xlabel('{\bf x1}');
        ylabel('{\bf x2}');
        titleText = ' Contour of $f(x) = c^Tx - \sum{log(b-Ax)}$';
        title(titleText,'Interpreter','latex');
        
        fprintf(' <strong> C.b   </strong> Plotting function f and its level sets');
        fprintf(' \n');  
    
    end 
    
    
    
%% EXERCISE C.c

    xk = zeros(n,1);
    x_GRADIENT(:,1) = xk;  
    
    alpha = 0.1;
    beta = 0.6;
    epsilon = 10^(-5);

    k = 1;
    t = 1;

    % Minimize f using gradient algorithm (backtracking line search)

    while ( norm( gradient_of_f(xk) ) ) > epsilon

        grad_f = gradient_of_f(xk); 
      
        % Step 1 
         Dxk = - grad_f;
 
        % Check if the new point, belongs to domf
         while(sum(b-A*(xk+t*Dxk) > 0) ~= length(b))
            t = beta*t;
         end            
            
        % Step 2 
         while (f(xk+t*Dxk) > f(xk) + alpha*t*(grad_f).'*Dxk)
            t = beta * t;
         end
    
        % Step 3 
         xk = xk + t * Dxk;
     
        % Step 4 
         k = k + 1;
                  
         x_GRADIENT(:,k+1) = xk;  
         f_GRADIENT(k+1) = f(xk); %#ok<CLALL>
         dif_GRADIENT(k+1) = f_GRADIENT(k+1) - p_star; %#ok<CLALL>
          
    end

    x_star_GRADIENT = xk;    
    p_star_GRADIENT = f(xk); 
 
    fprintf(' \n');  
    fprintf(' <strong> C.c   </strong> Optimal value (GRADIENT METHOD) = %4f \n',p_star_GRADIENT);
    fprintf('         Number of iterations in GRADIENT METHOD = %3d \n',k);
    fprintf(' \n');    
    
    
%% EXERCISE C.d

    xk = zeros(n,1) ; 
    kk = 1 ;

    % Calculate the Hessian
    hessian = 0;

     for i = (1:1:m)
         a_i = A(i,:).';
         hessian = hessian + (1/(b(i)-a_i.'*xk)).*(a_i*a_i.');
     end
 
    x_NEWTON(:,1) = xk;  

    condition = 1;
    
    while(condition)

       grad_f = gradient_of_f(xk); 

       Dxk = -inv(hessian)*grad_f ; 
       
       lamda = (grad_f).'*(-Dxk) ;

       % Break condition
       if( lamda <= 2*epsilon )
           condition = 0;
       else

           t =1 ;

           % Check if the new point belongs to the domf
           while(sum(b-A*(xk+t*Dxk) > 0) ~= m)
                t = beta*t;
           end

           % Backtracking line search
           while (f(xk + t*Dxk) > f(xk) + alpha*t*(grad_f).'*Dxk)
                t = beta*t ;
           end

           xk = xk +t*Dxk ;
           
           kk = kk+1 ;

           x_NEWTON(:,kk+1) = xk; 
           f_NEWTON(kk+1) = f(xk); %#ok<CLALL>
           dif_NEWTON(kk+1) = f_NEWTON(kk+1) - p_star; %#ok<CLALL>
           
           % Calculate the Hessian for the new point
           hessian = 0;

           for i = (1:1:m)
               a_i = A(i,:).';
               hessian = hessian + (1/(b(i)-a_i.'*xk)).*(a_i*a_i.');
           end

       end
       
     end     

    x_star_NEWTON = xk;    
    p_star_NEWTON = f(xk); 
 
    fprintf(' <strong> C.d   </strong> Optimal value (NEWTON METHOD) = %4f \n',p_star_NEWTON);
    fprintf('         Number of iterations in NEWTON METHOD = %3d \n',kk);
    fprintf(' \n');    
    
          
%% EXERCISE C.e
     
    figure();
    semilogy(0:k,dif_GRADIENT) ;
    hold on ;
    semilogy(0:kk,dif_NEWTON,'r') ;
    hold off ;
    legend('Gradient-Method with Backtracking Line Search','Newton-Method with Backtracking Line Search');     
    xlabel('k');
    ylabel('f-p*'); 
    titleText = '\bf {Plot f-p* with semilogy}';
    title(titleText,'Interpreter','latex'); 
   
     
     