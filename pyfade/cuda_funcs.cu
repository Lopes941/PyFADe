
extern "C" {

    
    __global__ void stomp_iteration(const double* series, 
                                    double* QT,
                                    const double* QT_old, 
                                    const double* means, 
                                    const double* stds, 
                                    double* MP, 
                                    size_t* inds_MP, 
                                    const double* QT_first, 
                                    const size_t i, 
                                    const size_t final_size, 
                                    const size_t interval_size,
                                    const size_t exclusion_zone_size,
                                    const size_t initial_size,
                                    const bool only_left){
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        double Dj, den;


        if (idx < final_size) {

            if (idx == 0){
                QT[idx] = QT_first[i];
            }else{
                QT[idx] =   QT_old[idx-1] 
                            - series[idx-1]*series[i-1] 
                            + series[idx+interval_size-1]*series[i+interval_size-1];
            }

            // if ( (idx>=i+exclusion_zone_size && idx >=initial_size) || ((idx>=i+exclusion_zone_size || idx<i-exclusion_zone_size) && ~only_left) ) {
            if (idx >=initial_size && (idx>=i+exclusion_zone_size || (idx<i-exclusion_zone_size && !only_left))){
            // if (idx>=i+exclusion_zone_size && idx >=initial_size) {

                den = interval_size*stds[i]*stds[idx];
                if (fabs(den) < 1e-9){
                    den = 1e-9;
                }
                Dj = sqrt(2 * interval_size * ( 1- (QT[idx] - interval_size*means[i]*means[idx] )/den ));

                if (MP[idx] > Dj){
                    MP[idx] = Dj;
                    inds_MP[idx] = i;
               }

            }
        }

    }

}