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


%% EXERCISE B.i
    
    n = 2; % n = 2, 50, 100, 1000

    K = 10; % K = 10, 100, 1000
        
    fprintf('You are using n = %d and K = %d. \n', n, K);
    fprintf('\n');
    
    
% Construct orthonormal matrix U

    A = rand(n,n);
    [U,S,V] = svd(A);
    
    matrix_UUT = U*U';
    matrix_UTU = U'*U;
    
    
% Construct matrix Ë
    
    Lmin = 1;
    Lmax = K * Lmin;
    
    random_L_mid = Lmin + (Lmax - Lmin) * rand(n - 2, 1);
    
    eigenvalues  = [Lmin; Lmax; random_L_mid];
    
    L = diag(eigenvalues);
    
    
    
%% EXERCISE B.ii

 % Construct random vector q

  q = rand(n,1);

  
 % Construct positive definite matrix P
  
  P = U*L*U';

  
  
%% EXERCISE B.iii

% Closed-form solution 
    
    x_star = - P \ q;
    
    p_star = - 0.5 * q' * ( P \ q );
    
    fprintf(' <strong> B.iii </strong> Optimal value (closed-form solution) = %4f \n',p_star);
    fprintf(' \n');
    
    
    
%% EXERCISE B.iv 

  % Function f
  f = @(x) 0.5*x'*P*x + q'*x;
  
  
  % CVX solution
   
    cvx_begin quiet
        variable x(n)
        minimize( f(x) )
    cvx_end
    
    x_star_cvx = x;
    
    
    
%% EXERCISE B.v


 % --------------------- Exact line search ---------------------
    
    epsilon = 10^(-5);
    
    % Initial point x_0 is random
    x_0 = 10*randn(n,1);
    xk = x_0;
    
    k = 0;
    
    x_exact(:,1) = xk;  
    f_exact(1) = f(xk);
    dif_exact(1) = f_exact(1) - p_star;  
    
    
    while (norm(P*xk + q)  > epsilon)  
      
        grad_f = P*xk + q; 
        
        
        % Step 1 : Äx_k = - grad(f(x_k)) 

            Dxk = - grad_f;
        
            
        % Step 2 : Line search and choose t*
    
            t_star = (norm(grad_f))^2 / ((grad_f)' * P * grad_f);
            
            
        % Step 3 : x(k+1) = x(k) + t_star * Äx_k
        
            xk = xk + t_star * Dxk;
    
      
        % Step 4 : k = k + 1
    
            k = k + 1;
  
        % FOR EXERCISE B.vii
   
            x_exact(:,k+1) = xk;  
            f_exact(k+1) = f(xk); %#ok<SAGROW>
            dif_exact(k+1) = f_exact(k+1) - p_star;        %#ok<SAGROW>
      
    end
    
    x_star_exact = xk;
    
    fprintf(' <strong> B.v   </strong> Optimal value (exact line search) = %4f \n',f(x_star_exact));
    fprintf('         Number of iterations in exact line search = %3d \n',k);
    fprintf(' \n');
    
    
    
    % ----------------- Backtracking line search -----------------
    
    
    % Initial point x is random
  
    xkk = x_0;
 
    t = 1;
    alpha = 0.25; 
    beta  = 0.60;

    kk = 0;
  
    x_back(:,1) = xkk;  
    f_back(1) = f(xkk);
    dif_back(1) = f_back(1) - p_star;
    
    while (norm(P*xkk + q)) > epsilon

        grad_f = P*xkk + q; 
      
        % Step 1 : Äx_kk = - grad(f(x_kk)) 
            
            Dxkk = - grad_f;
    
        % Step 2 :
        
            while (f(xkk+t*Dxkk) > f(xkk) + alpha*t*(grad_f).'*Dxkk)

                t = beta * t;

            end
    
        % Step 3 : x(k+1) = x(k) + t_star * Dxk
    
            xkk = xkk + t * Dxkk;
     
   
        % Step 4 : k = k + 1
     
            kk = kk + 1;
     
     
        % FOR EXERCISE B.vii
     
        x_back(:,kk+1) = xkk;  
        f_back(kk+1) = f(xkk); %#ok<SAGROW>
        dif_back(kk+1) = f_back(kk+1) - p_star; %#ok<SAGROW>
         
     
        % UNCOMMENT THE FOLLOWING LINE IF YOU WANT TO VIEW EACH ITERATION
        % fprintf('Iterations =%3d --> Optimal value = %4f \n',kk, f(xkk,P,q));
    
    
    end
    
    x_star_back = xkk;
    
    fprintf(' <strong> B.v   </strong> Optimal value (backtracking line search) = %4f \n',f(x_star_back));
    fprintf('         Number of iterations in backtracking line search = %3d \n',kk);
    fprintf(' \n');
 
  
%% EXERCISE B.vi

    %  Make sure that n = 2. Contour plot f(xk)
    if (n == 2)
        
        start_1 = max([abs(x_back(1,:)) abs(x_exact(1,:))]);
        start_2 = max([abs(x_back(2,:)) abs(x_exact(2,:))]);
        
        [x1,x2] = meshgrid( xk(1)-start_1-5 : 0.1 : xk(1)+start_1+5 , xk(2)-start_2-5 : 0.1 : xk(2)+start_2+5 );

        func = zeros(size(x1));

        for i = 1 : size(x1,1)
            
            for j = 1 : size(x2,2)
                
                X = [x1(i,j) ; x2(i,j)];
                
                func(i,j) = f(X);

            end
            
        end
        
        figure();
        
        plot(x_exact(1,:),x_exact(2,:),'-r*'); 
        hold on;
        plot(x_back(1,:),x_back(2,:),'-ko');
        contour(x1,x2,func,30);
        colorbar;
        legend('Exact line search','Backtracking line search','Levels');    
        legend('Location','southeast')
        xlabel('\bf {x1}');
        ylabel('\bf {x2}'); 
        titleText = ['\bf {Contour plot of f(xk) for K = }',num2str(K)];
        title(titleText,'Interpreter','latex');
        hold off;
    
    end     
    

    
%% EXERCISE B.vii
    
    % Plot quatity log(f(x_k) - p_star) for the 2 algorithms
  
    figure();
    plot(0:k,log(dif_exact),'-r*');
    hold on;
    plot(0:kk,log(dif_back),'-bo');
    legend('Exact line search','Backtracking line search'); 
    grid on;
    xlabel('k');
    ylabel('Log(f-p*)'); 
    titleText = ['\bf {Plot log(f-p*) for K=}',num2str(K)];
    title(titleText,'Interpreter','latex');  
    
    
    

%% EXERCISE B.viii

    c_exact = 1 - 1/K;
    k_min_exact = ceil(log((f_exact(1)-p_star)/epsilon)/log(1/c_exact));
    
    if(k_min_exact == 0)
        k_min_exact = 1;
    end
    
    c_back = 1 - min(2*alpha, 2*alpha*beta/K);
    k_min_back = ceil(log((f_back(1)-p_star)/epsilon)/log(1/c_back));

    if(k_min_back == 0)
        k_min_back = 1;
    end
        

    fprintf(' <strong> B.viii   </strong> Minimum theoretical iterations (exact line search) = %d \n',k_min_exact);
    fprintf(' <strong>          </strong> Minimum theoretical iterations (backtracking line search) = %d \n',k_min_back);
    fprintf(' \n'); 
  






