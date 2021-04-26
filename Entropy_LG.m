clear
close all
clc

%% Initializations

lg_exp_gamma = -10 : 0.1 : 10;
cov_spread = -3 : 0.01 : 10;
gamma = 0.005 : 0.001 : 0.995;
entropy = zeros( length( gamma ) , length( cov_spread ) );

%% Computation

for idx_exp = 1 : 1 : length( gamma )
    for idx_cov = 1 : 1 : length( cov_spread )     
        for idx_gamma = 1 : 1 : length( gamma )
            
            covariance = exp( cov_spread( idx_cov ) );
            %covariance = cov_spread( idx_cov );
            
%             pdf_value = 1 / sqrt( 2 * pi * covariance )...
%                 / gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) *...
%                 exp( -0.5 * ( log( gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) )...
%                 - lg_exp_gamma( idx_exp ) ) ^2 / covariance );
            
            pdf_value = 1 / sqrt( 2 * pi * covariance )...
                / gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) *...
                exp( -0.5 * ( log( gamma( idx_gamma ) / ( 1 - gamma( idx_gamma ) ) )...
                - log( gamma( idx_exp ) / ( 1 - gamma( idx_exp ) ) ) ) ^2 / covariance );
            
            if pdf_value > 0
                entropy( idx_exp , idx_cov ) = entropy( idx_exp , idx_cov ) - pdf_value * log( pdf_value ) / length( gamma );
            end
            
        end
    end
end

%% Figures
%[ exp_mesh , cov_mesh ] = meshgrid( exp( lg_exp_gamma ) ./ ( 1 + exp( lg_exp_gamma ) ) , cov_spread );
[ exp_mesh , cov_mesh ] = meshgrid( log( gamma ./ ( 1 - gamma ) ) , exp( cov_spread ) );

figure( 'Name', 'Entropy Figure' )
grid on
h = surf( exp_mesh' , cov_mesh' , entropy );
set(gca,'FontSize',18,'yscale' , 'log')
set( h , 'edgecolor' , 'none')
xlabel('E[l\lambda]')
ylabel('Var(l\lambda)')

return;
