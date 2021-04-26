clear
close all
clc

%% Initialization
x = 0 : 1 : 360;
gamma = 0.002 : 0.002: 0.998;
y_1_logit = 0.3 * cos(2 * x * pi / 180) + 0.3;
y_2_logit = - 0.3 - 0.3 * cos(2 * x * pi / 180);
y_1 = exp(y_1_logit) ./ (1 + exp(y_1_logit));
y_2 = exp(y_2_logit) ./ (1 + exp(y_2_logit));
R_1 = 2 * (0.7 + 0.3 * cos(x * pi / 180));
R_2 = 2 * (0.7 - 0.3 * cos(x * pi / 180));
pdf_value_1 = zeros( length(x) , length(gamma) );
pdf_value_2 = zeros( length(x) , length(gamma) );
phi_1 = zeros( 1, length(x) );
phi_2 = zeros( 1, length(x) );
covariance_1 = zeros( 1, length(x) );
covariance_2 = zeros( 1, length(x) );

%% Computation
for idx_x = 1 : 1 : length(x)
    for idx_gamma = 1 : 1 : length(gamma)
        
        covariance_1(idx_x) = 1 / R_1( idx_x )^2;
        covariance_2(idx_x) = 1 / R_2( idx_x )^2;
        pdf_value_1( idx_x , idx_gamma ) = 1 / sqrt( 2 * pi * covariance_1(idx_x) )...
            / gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) *...
            exp( -0.5 * ( log( gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) )...
            - y_1_logit( idx_x ) )^2 / covariance_1(idx_x) );
        pdf_value_2( idx_x , idx_gamma ) = 1 / sqrt( 2 * pi * covariance_2(idx_x) )...
            / gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) *...
            exp( -0.5 * ( log( gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) )...
            - y_2_logit( idx_x ) )^2 / covariance_2(idx_x) );
        phi_1( idx_x ) = ( y_1_logit( idx_x ) * covariance_1(idx_x)^-1 - y_2_logit( idx_x ) * covariance_2(idx_x)^-1 );
        phi_2( idx_x ) = ( y_1_logit( idx_x ) ^ 2 * covariance_1(idx_x)^-1 - y_2_logit( idx_x ) ^ 2 * covariance_2(idx_x)^-1 ); 
        
    end
end

%% Graphs
[ x_mesh, gamma_mesh ] = meshgrid(x, gamma);

figure('Name', 'PDF value of logg, model 1')
grid on
h = surf( x_mesh' , gamma_mesh' , pdf_value_1 );
set( h , 'edgecolor', 'none')
set(gca, 'FontSize' , 18)
xlabel('\psi[deg]')
xticks([0, 90, 180, 270, 360])
ylabel('\gamma')
axis([0 360 0 1])

figure('Name', 'PDF value of logg, model 2')
grid on
h = surf( x_mesh' , gamma_mesh' , pdf_value_2 );
set( h , 'edgecolor', 'none')
set(gca, 'FontSize' , 18)
xlabel('\psi[deg]')
xticks([0, 90, 180, 270, 360])
ylabel('\gamma')
axis([0 360 0 1])

figure('Name', 'Phi_1 value')
grid on
plot(x, phi_1), grid
xlabel('x^{rel}')
ylabel('\Phi_1')

figure('Name', 'Phi_2 value')
grid on
plot(x, phi_2), grid
xlabel('x^{rel}')
ylabel('\Phi_2')

%% PDF histograms
lgamma_data = normrnd(0.6, sqrt(0.25), [10000000, 1]);
data_point = 1;

figure('Name', 'l\gamma histogram')
h = histogram(lgamma_data);
h.BinWidth = 0.1;
set(gca, 'FontSize', 18)
grid on
xlabel('l\gamma value')
ylabel('Number of appearances')

figure('Name', 'JLP likelihood')
grid on
histogram(lgamma_data * phi_1(data_point) - 0.5 * phi_2(data_point));
set(gca, 'FontSize', 18)
xlabel('Likelihood value')
ylabel('Number of appearances')

likelihood_accurate = - 0.5 * log(covariance_1(data_point)) + 0.5 * log(covariance_2(data_point))...
    - 0.5 * lgamma_data.^2 * covariance_1(data_point)^-1 + 0.5 * lgamma_data.^2 * covariance_2(data_point)^-1 ...
    + lgamma_data * phi_1(data_point) - 0.5 * phi_2(data_point);

series_vector = (1 : 1e6) * 1e-4 - 50;
transformation_check = - 0.5 * log(covariance_1(data_point)) + 0.5 * log(covariance_2(data_point))...
    - 0.5 * series_vector.^2 * covariance_1(data_point)^-1 + 0.5 * series_vector.^2 * covariance_2(data_point)^-1 ...
    + series_vector * phi_1(data_point) - 0.5 * phi_2(data_point);

figure('Name', 'Real likelihood')
hold on
h1 = histogram(likelihood_accurate);
h2 = histogram(lgamma_data * phi_1(data_point) - 0.5 * phi_2(data_point));
h3 = histogram(lgamma_data);
h1.BinWidth = 0.001;
h1.EdgeColor = 'None';
h2.BinWidth = 0.001;
h2.EdgeColor = 'None';
h3.BinWidth = 0.001;
h3.EdgeColor = 'None';
legend('Real', 'Approximated', 'l \gamma value')
set(gca, 'FontSize', 18, 'XTick', [-5, -2.5, 0, 2.5, 5], 'YTicklabel', [])
grid on
xlabel('Likelihood value')
ylabel('PDF value')
xlim([-5 5])
ylim([0, 1.5e4])

figure('Name', 'Transformation Check')
grid on
plot(series_vector, transformation_check);
set(gca, 'FontSize', 18)
xlabel('Transformation Value')
ylabel('PDF Value')

return;