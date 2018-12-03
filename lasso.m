%%  IMPLEMENTAÇÃO DE LA SSO COM ALGORITMO DE COODENADA DESCENDENTE CICLICA %%

clear all;
close all;
clc;


%% SETUP %%
N = 50;                % numero de observações
P = 25;                % numero de preditores 
k = 15;                % numero de elementos diferente de zero do vetor Beta original
m = 50;               % numero de ciclos
nivel_ruido = 0.01;    % nivel de ruido
beta = zeros(P, 1);    % vetor dos coeficientes da regressão real
n_lambda = 300;        % numero de lambdas


%% CONSTRUINDO O CONJUNTO DE DADOS %%

% DADOS DE ENTRADA - CADA LINHA É UMA VETOR AMOSTRA xi 
X = randn(N,P);                                      % matriz [N x p] de distribuição normal

% PESOS REAIS 
index = randperm(P);                                 % gera vetor de 1 á N embaralhado
index = index(1:k);                                  % seleciona as localizações diferente de zero   
beta(index) = randn(k,1);                            % valores de beta real 

% DADOS DE SAIDA
Y = X*beta + nivel_ruido*randn(N,1);                 % adicionando ruido aleatorio           

% calculo do lambda (1º teste)
% lambda_max = max(abs(X'*Y));                    % maior valor do vetor beta ideal
% lambdas = linspace(0, lambda_max, n_lambda);  % vetor de lambdas  

% PADRONIZANDO as colunas da matriz X
for i = 1:P,
    X(:,i) = X(:,i) - mean(X(:,i));
end

% PADRONIZANDO o vetor de saida Y.
Y = Y - mean(Y);

% calculo do lambda (2º teste) 
lambda_max = max(abs(X'*Y));                        % maior valor do vetor beta ideal
lambdas = linspace(0, lambda_max, n_lambda);        % vetor de lambdas  

   

%% ALGORITMO DO SUBGRADIENTE CYCLICO %%

mt_betas = zeros(n_lambda, P);      % calcula os valores do peso aleatoriamente 

% laço percorre os lambdas
for l = 1:n_lambda
    
    % valores iniciais
    B = randn(P, 1);           % vetor de pesos B calculado pelo algoritmo do Sub-gradiente iniciado zerado (LS, randn)
    % R = Y - X*B;             % calculo do residuo para B inicial
    
    % COORDENADA DESCENDENTE CICLICA
    
    % numero de ciclos
    for k = 1:m, 

        % calculando o sub-gradiente p/ cada coordenada
        for j = 1:P,

            % calculando o residuo para a coordenada
            R = Y - X*B + B(j)*X(:,j);

            pj = (1/N)*dot(X(:,j), R);

            % configurando o peso Bj
            if pj > lambdas(l)
                B(j) = pj - lambdas(l);
            end

            if pj < -lambdas(l)
                B(j) = pj + lambdas(l);   
            end

            if (pj <= lambdas(l)) && (pj >= -lambdas(l))
                B(j) = 0;    
            end

        end     % fim do for p/ j. 
        
    end    % fim do for p/ k

   mt_betas(l,:) = B';
   
end        % fim do for p/ l

plot(lambdas', mt_betas)
xlim([0 3]);
xlabel('lambda (\lambda)');             % titulo do eixo-x
ylabel('Coeficientes');                 % titulo do eixo-y
title('LASSO PARA REGRESSÃO SIMPLES')   % Titulo do grafico

% plotar os vetores de Beta em relação ao maior beta de cada vetor
denominador = sum(abs(mt_betas(1,:)));
pct = sum(abs(mt_betas),2)/denominador;
figure;
plot(pct, mt_betas);
xlabel('\beta / \beta_{max}');             % titulo do eixo-x
ylabel('Coeficientes');                 % titulo do eixo-y
title('LASSO PARA REGRESSÃO SIMPLES')   % Titulo do grafico



