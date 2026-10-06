#include "config.h"
#include "fp_compression.hpp"
#include "data_types.hpp"


namespace hmat{

/**
 * Instanciate FP compressors depending on the chosen method. If the method is unknown, undefined or not included, return a default compressor (which does not compress)
 */
template<typename T>
FPCompressorInterface<T>* initCompressor(hmat_FPcompress_t method)
{
    FPCompressorInterface<T>* res;

    switch (method)
        {
    #ifdef HAVE_COMPOSYX

    #ifdef COMPOSYX_USE_ZFP_COMPRESSOR

            
        case ZFP_COMPRESSOR:
            res = new ZFPcompressor<T>();

            break;
    #endif //COMPOSYX_USE_ZFP_COMPRESSOR

    #ifdef COMPOSYX_USE_SZ3_COMPRESSOR
        case SZ3_COMPRESSOR:
            res = new SZ3compressor<T>();

            break;

    #endif //COMPOSYX_USE_SZ3_COMPRESSOR

    #ifdef COMPOSYX_USE_SZ_COMPRESSOR

        case SZ_COMPRESSOR:
            res = new SZcompressor<T>();
            break;
    #endif //COMPOSYX_USE_SZ_COMPRESSOR

    #ifdef COMPOSYX_USE_BLOSC2_COMPRESSOR

        case BLOSC2_COMPRESSOR:
            res = new BLOSC2compressor<T>();
            break;
    #endif //COMPOSYX_USE_BLOSC2_COMPRESSOR
        
    #endif //HAVE_COMPOSYX


            
        case DEFAULT_COMPRESSOR:          
        default:
            res = new Defaultcompressor<T>();
        }

    return res;
}


template<typename T>
void Defaultcompressor<T>::compress(T* data, size_t size, double epsilon)
{
    //unit roundoffs
    constexpr double float_unit_roundoff = std::numeric_limits<float>::epsilon(); // ~1.19e-7
    constexpr double half_unit_roundoff  = 9.77e-4;

    //Computing the max absolute value
    double max_abs = 0.0;
    
    for (size_t i = 0; i < size; ++i) {
        // std::abs fonctionne nativement pour float, double, ET std::complex
        double current_abs = std::abs(data[i]); 
        if (current_abs > max_abs) {
            max_abs = current_abs;
        }
    }
    
   
    //Maximum absolute error induced by compression
    double eps_rel = (max_abs == 0.0) 
            ? std::numeric_limits<double>::infinity() 
            : (epsilon / max_abs);

    // --- Step A: Check if we should keep Double precision ---
    // Evaluated at compile-time: Is T a double-precision type?
    if constexpr (!std::is_same_v<T, SinglePrecType>) {
        if (eps_rel < float_unit_roundoff) {
            // Epsilon is very strict, keep original high precision
            if (std::holds_alternative<std::vector<T>>(_data)) {
                std::get<std::vector<T>>(_data).assign(data, data + size);
            } else {
                _data = std::vector<T>(data, data + size);
            }
            _ratio = 1.0;
            return;
        }
    }

    // --- Step B: Check if we should drop to (or keep) Single precision ---
    // Evaluated at compile-time: Is Single better than Half for this type?
    if constexpr (!std::is_same_v<SinglePrecType, HalfPrecType>) {
        if (eps_rel < half_unit_roundoff) {
            // Epsilon allows Single precision. 
            // The vector constructor automatically casts T -> SinglePrecType (e.g., double -> float)
            if (std::holds_alternative<std::vector<SinglePrecType>>(_data)) {
                std::get<std::vector<SinglePrecType>>(_data).assign(data, data + size);
            } else {
                _data = std::vector<SinglePrecType>(data, data + size);
            }
            _ratio = static_cast<double>(sizeof(T)) / sizeof(SinglePrecType);
            return;
        }
    }

    // --- Step C: Default fallback to Half precision ---
    // Epsilon is large enough, drop to minimum precision.
    // The vector constructor automatically casts T -> HalfPrecType
    // using the custom constructors we wrote in Step 1!
    if (std::holds_alternative<std::vector<HalfPrecType>>(_data)) {
        std::get<std::vector<HalfPrecType>>(_data).assign(data, data + size);
    } else {
        _data = std::vector<HalfPrecType>(data, data + size);
    }
    _ratio = static_cast<double>(sizeof(T)) / sizeof(HalfPrecType);
}

template<typename T>
std::vector<T> Defaultcompressor<T>::decompress()
{
    std::vector<T> out = decompressCopy();
    std::visit([](auto& vec) { vec.clear(); }, _data);
    _ratio = 1; //ratio is reset
    return out;
}

template <typename T>
void Defaultcompressor<T>::decompress(T *dest)
{
    decompressCopy(dest);
    std::visit([](auto& vec) { vec.clear(); }, _data);
    _ratio = 1; //Ratio is reset
}

template <typename T>
std::vector<T> Defaultcompressor<T>::decompressCopy()
{
    return std::visit([](auto& vec) {
        return std::vector<T>(vec.begin(), vec.end());
    }, _data);
}

template <typename T>
void Defaultcompressor<T>::decompressCopy(T *dest)
{   
    std::visit([dest](auto& vec) {
        std::copy(vec.begin(), vec.end(), dest);
    }, _data);
}

template <typename T>
double Defaultcompressor<T>::get_ratio()
{
    return _ratio;
}


#ifdef HAVE_COMPOSYX


#ifdef COMPOSYX_USE_SZ_COMPRESSOR

template <typename T>
SZcompressor<T>::~SZcompressor()
{
    if(_compressor)
    {
        delete _compressor;
        _compressor = nullptr;
    }
}

template <typename T>
void SZcompressor<T>::compress(T* data, size_t size, double epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    
    this->_size = size;
    this->_compressor = new composyx::SZ_compressor<T, composyx::SZ_CompressionMode::ABSOLUTE>(data, size, epsilon);
}

template<typename T>
std::vector<T> SZcompressor<T>::decompress()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::vector<T> out = decompressCopy();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
void SZcompressor<T>::decompress(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    decompressCopy(dest);
    delete _compressor;
    _compressor = nullptr;
}

template <typename T>
std::vector<T> SZcompressor<T>::decompressCopy()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _compressor->decompress();
}

template <typename T>
void SZcompressor<T>::decompressCopy(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _compressor->decompress(dest);
}

template <typename T>
double SZcompressor<T>::get_ratio()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n",sizeof(T)* this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return this->_compressor->get_ratio();
}

#endif //COMPOSYX_USE_SZ_COMPRESSOR


#ifdef COMPOSYX_USE_SZ3_COMPRESSOR

template <typename T>
SZ3compressor<T>::~SZ3compressor()
{
     if(_compressor)
    {
         delete _compressor;
        _compressor = nullptr;
    }
}

template <typename T>
void SZ3compressor<T>::compress(T* data, size_t size, double epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    this->_size = size;
    this->_compressor = new composyx::SZ3_compressor<T, SZ3::EB::EB_ABS>(data, size, epsilon);
}

template<typename T>
std::vector<T> SZ3compressor<T>::decompress()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::vector<T> out = decompressCopy();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
void SZ3compressor<T>::decompress(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    decompressCopy(dest);
    delete _compressor;
    _compressor = nullptr;
}

template <typename T>
std::vector<T> SZ3compressor<T>::decompressCopy()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _compressor->decompress();
}

template <typename T>
void SZ3compressor<T>::decompressCopy(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _compressor->decompress(dest);
}

template <typename T>
double SZ3compressor<T>::get_ratio()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return this->_compressor->get_ratio();
}


#endif //COMPOSYX_USE_SZ3_COMPRESSOR

#ifdef COMPOSYX_USE_ZFP_COMPRESSOR

template <typename T>
ZFPcompressor<T>::~ZFPcompressor()
{
     if(_compressor)
    {
         delete _compressor;
        _compressor = nullptr;
    }
}

template <typename T>
void ZFPcompressor<T>::compress(T* data, size_t size, double epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    this->_size = size;
    this->_compressor = new composyx::ZFP_compressor<T, composyx::ZFP_CompressionMode::ACCURACY>(data, size, epsilon);
}

template<typename T>
std::vector<T> ZFPcompressor<T>::decompress()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::vector<T> out = decompressCopy();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
void ZFPcompressor<T>::decompress(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    decompressCopy(dest);
    delete _compressor;
    _compressor = nullptr;
}

template <typename T>
std::vector<T> ZFPcompressor<T>::decompressCopy()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _compressor->decompress();
}

template <typename T>
void ZFPcompressor<T>::decompressCopy(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _compressor->decompress(dest);
}

template <typename T>
double ZFPcompressor<T>::get_ratio()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    return this->_compressor->get_ratio();
}

#endif //COMPOSYX_USE_ZFP_COMPRESSOR


#ifdef COMPOSYX_USE_BLOSC2_COMPRESSOR

template <typename T>
BLOSC2compressor<T>::~BLOSC2compressor()
{
     if(_compressor)
    {
         delete _compressor;
        _compressor = nullptr;
    }
}

template <typename T>
void BLOSC2compressor<T>::compress(T* data, size_t size, double epsilon)
{
    //printf("BLOSC2 Compress\n");
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    double max = 0.0;
    for(int i = 0; i < size; i++)
    {
        if (std::abs(data[i]) > max)
        {
            max = std::abs(data[i]);
        }
    }
    //preventing division by 0
    if (max < 1e-300) {
        max = 1.0; 
    }
    double zeta = epsilon / max;

    this->_size = size;
    this->_compressor = new composyx::Blosc2_compressor<T>(data, size, zeta);
}

template<typename T>
std::vector<T> BLOSC2compressor<T>::decompress()
{
    //printf("BLOSC2 Decompress\n");
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::vector<T> out = decompressCopy();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
void BLOSC2compressor<T>::decompress(T *dest)
{
   // printf("BLOSC2 Decompress\n");
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    decompressCopy(dest);
    delete _compressor;
    _compressor = nullptr;
}

template <typename T>
std::vector<T> BLOSC2compressor<T>::decompressCopy()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _compressor->decompress();
}

template <typename T>
void BLOSC2compressor<T>::decompressCopy(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _compressor->decompress(dest);
}

template <typename T>
double BLOSC2compressor<T>::get_ratio()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    return this->_compressor->get_ratio();
}


#endif //COMPOSYX_USE_BLOSC2_COMPRESSOR


#endif // HAVE_COMPOSYX



// Templates declaration
template FPCompressorInterface<S_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<D_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<C_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<Z_t>* initCompressor(hmat_FPcompress_t method);

}

