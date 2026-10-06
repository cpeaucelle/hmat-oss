#pragma once
#include <vector>
#include <mutex>
#include <complex>
#include <cstdint>
#include <cstring>

#include "rk_matrix.hpp"





#ifdef HAVE_COMPOSYX

#include "composyx.hpp"
#include "composyx/interfaces/basic_concepts.hpp"
#include "composyx/utils/Arithmetic.hpp"

#ifdef COMPOSYX_USE_ZFP_COMPRESSOR
#include "composyx/utils/ZFP_compressor.hpp"
#endif //COMPOSYX_USE_ZFP_COMPRESSOR

#ifdef COMPOSYX_USE_SZ_COMPRESSOR
#include "composyx/utils/SZ_compressor.hpp"
#endif //COMPOSYX_USE_SZ_COMPRESSOR

#ifdef COMPOSYX_USE_SZ3_COMPRESSOR
#include "composyx/utils/SZ3_compressor.hpp"
#endif //COMPOSYX_USE_SZ3_COMPRESSOR

#ifdef COMPOSYX_USE_BLOSC2_COMPRESSOR
#include "composyx/utils/Blosc2_compressor.hpp"
#endif //COMPOSYX_USE_BLOSC2_COMPRESSOR


#endif //HAVE_COMPOSYX


// =====================================================================
// 1. Emulation of float16_t with uint16_t
// =====================================================================
struct HalfFloat {
    uint16_t value;

    // --- Binary conversion utilities (IEEE-754) ---
    static uint16_t float_to_half_bits(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(float));
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
        uint32_t mant = x & 0x007fffff;

        if (exp <= 0) return sign; // Simplification: flush denormals to zero
        if (exp >= 31) return sign | 0x7c00 | (mant ? 1 : 0); // Infinity and NaN
        return sign | (exp << 10) | (mant >> 13); // Normal numbers
    }

    static float half_bits_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = (h & 0x7c00) >> 10;
        uint32_t mant = h & 0x03ff;

        uint32_t x = 0;
        if (exp == 0 && mant == 0) x = sign;
        else if (exp == 31) x = sign | 0x7f800000 | (mant << 13);
        else x = sign | ((exp - 15 + 127) << 23) | (mant << 13);

        float f;
        std::memcpy(&f, &x, sizeof(float));
        return f;
    }

public:
    // --- Constructors ---
    HalfFloat() = default; // Required for std::vector
    
    // Constructor from float (binary conversion)
    HalfFloat(float f) : value(float_to_half_bits(f)) {}
    
    // Constructor from double (routes through float)
    HalfFloat(double d) : HalfFloat(static_cast<float>(d)) {}

    // --- Conversion operators ---
    // Conversion to float
    operator float() const { return half_bits_to_float(value); }
    
    // Conversion to double (routes through float)
    operator double() const { return static_cast<double>(half_bits_to_float(value)); }
};


// =====================================================================
// 2. Emulation of complex<float16>
// =====================================================================
struct ComplexHalf {
    HalfFloat real;
    HalfFloat imag;

public:
    // --- Constructors ---
    ComplexHalf() = default; // Required for std::vector
    
    // Constructor from std::complex<float>
    ComplexHalf(std::complex<float> c) : real(c.real()), imag(c.imag()) {}
    
    // Constructor from std::complex<double> (delegates to HalfFloat(double))
    ComplexHalf(std::complex<double> c) : real(c.real()), imag(c.imag()) {}

    // --- Conversion operators ---
    // Conversion to std::complex<float>
    operator std::complex<float>() const { 
        return std::complex<float>(static_cast<float>(real), static_cast<float>(imag)); 
    }
    
    // Conversion to std::complex<double>
    operator std::complex<double>() const { 
        return std::complex<double>(static_cast<double>(real), static_cast<double>(imag)); 
    }
};

// =====================================================================
// Type Traits: Mapping original types to compressed equivalents
// =====================================================================

// --- 1. Single Precision (32-bit) Traits ---
template <typename T> struct SinglePrecision;

template <> struct SinglePrecision<float>                { using type = float; };
template <> struct SinglePrecision<double>               { using type = float; };
template <> struct SinglePrecision<std::complex<float>>  { using type = std::complex<float>; };
template <> struct SinglePrecision<std::complex<double>> { using type = std::complex<float>; };

template <typename T>
using SinglePrecision_t = typename SinglePrecision<T>::type;


// --- 2. Half Precision (16-bit) Traits ---
template <typename T> struct HalfPrecision;

template <> struct HalfPrecision<float>                { using type = HalfFloat; };
template <> struct HalfPrecision<double>               { using type = HalfFloat; };
template <> struct HalfPrecision<std::complex<float>>  { using type = ComplexHalf; };
template <> struct HalfPrecision<std::complex<double>> { using type = ComplexHalf; };

template <typename T>
using HalfPrecision_t = typename HalfPrecision<T>::type;


namespace hmat 
{


#ifdef HAVE_COMPOSYX

#ifdef COMPOSYX_USE_SZ_COMPRESSOR
template<typename T>
class SZcompressor : public FPCompressorInterface<T> {
private:
    composyx::SZ_compressor<T, composyx::SZ_CompressionMode::ABSOLUTE>* _compressor;
    size_t _size;

    mutable std::recursive_mutex _mutex;

public:
    SZcompressor() : _compressor(nullptr), _size(0) {};

    ~SZcompressor();

    void compress(T* data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    void decompress(T* dest) override;

    std::vector<T> decompressCopy() override;

    void decompressCopy(T* dest) override;

    double get_ratio() override;

    SZcompressor* copy() override {
      std::lock_guard<std::recursive_mutex> lock(_mutex);
      
      SZcompressor* newComp = new SZcompressor();
      newComp->_size = _size;
      //TODO : implement deepcopy method in composyx
      //newComp->_compressor = new composyx::SZ_compressor<T, composyx::SZ_CompressionMode::POINTWISE>(*_compressor);
      return newComp;
    }

    
};
#endif //COMPOSYX_USE_SZ_COMPRESSOR

#ifdef COMPOSYX_USE_SZ3_COMPRESSOR

template<typename T>
class SZ3compressor : public FPCompressorInterface<T> {
private:
    composyx::SZ3_compressor<T, SZ3::EB::EB_ABS>* _compressor;
    size_t _size;

    mutable std::recursive_mutex _mutex;

public:
    SZ3compressor() : _compressor(nullptr), _size(0) {};

    ~SZ3compressor();

    void compress(T* data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    void decompress(T* dest) override;

    std::vector<T> decompressCopy() override;

    void decompressCopy(T* dest) override;

    double get_ratio() override;

    SZ3compressor* copy() override {
      std::lock_guard<std::recursive_mutex> lock(_mutex);

      SZ3compressor* newComp = new SZ3compressor();
      newComp->_size = _size;
      //TODO : implement deepcopy method in composyx
      //newComp->_compressor = new composyx::SZ3_compressor<T, SZ3::EB::EB_REL>(*_compressor);
      return newComp;
    }

    
};

#endif //COMPOSYX_USE_SZ3_COMPRESSOR

#ifdef COMPOSYX_USE_ZFP_COMPRESSOR

template<typename T>
class ZFPcompressor : public FPCompressorInterface<T> {
private:
    composyx::ZFP_compressor<T, composyx::ZFP_CompressionMode::ACCURACY>* _compressor;
    size_t _size;

    mutable std::recursive_mutex _mutex;

public:
    ZFPcompressor() : _compressor(nullptr), _size(0) {};

    ~ZFPcompressor();

    void compress(T* data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    void decompress(T* dest) override;

    std::vector<T> decompressCopy() override;

    void decompressCopy(T* dest) override;

    double get_ratio() override;

    ZFPcompressor* copy() override {
      std::lock_guard<std::recursive_mutex> lock(_mutex);
      
      ZFPcompressor* newComp = new ZFPcompressor();
      newComp->_size = _size;
      //TODO : implement deepcopy method in composyx
      //newComp->_compressor = _compressor->copy();
      return newComp;
    }

    
};

#endif //COMPOSYX_USE_ZFP_COMPRESSOR


#ifdef COMPOSYX_USE_BLOSC2_COMPRESSOR

template<typename T>
class BLOSC2compressor : public FPCompressorInterface<T> {
private:
    composyx::Blosc2_compressor<T>* _compressor;
    size_t _size;

    mutable std::recursive_mutex _mutex;

public:
    BLOSC2compressor() : _compressor(nullptr), _size(0) {};

    ~BLOSC2compressor();

    void compress(T* data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    void decompress(T* dest) override;

    std::vector<T> decompressCopy() override;

    void decompressCopy(T* dest) override;

    double get_ratio() override;

    BLOSC2compressor* copy() override {
      std::lock_guard<std::recursive_mutex> lock(_mutex);
      
      BLOSC2compressor* newComp = new BLOSC2compressor();
      newComp->_size = _size;
      //TODO : implement deepcopy method in composyx
      //newComp->_compressor = _compressor->copy();
      return newComp;
    }

    
};

#endif //COMPOSYX_USE_BLOSC2_COMPRESSOR


#endif // HAVE_COMPOSYX


/* Default compressor if composyx is not installed
*/
template<typename T>
class Defaultcompressor : public FPCompressorInterface<T> {
private:
    using HalfPrecType = HalfPrecision_t<T>;
    using SinglePrecType = SinglePrecision_t<T>;

    // --- Dynamic Variant Generation ---
    // We use nested std::conditional_t to prevent duplicate types in the variant.
    // Duplicate types would make assignment and std::visit ambiguous.
    using StorageVariant = std::conditional_t<
        // Condition 1: Is T the same as Single precision? (e.g., T is float)
        std::is_same_v<T, SinglePrecType>,
        
        // IF TRUE (T is Single or lower):
        std::conditional_t<
            // Condition 2: Is T also the same as Half precision? 
            std::is_same_v<T, HalfPrecType>,
            std::variant<std::vector<HalfPrecType>>,                                 // Only 1 type
            std::variant<std::vector<HalfPrecType>, std::vector<T>>                  // 2 types (Half, Single)
        >,
        
        // IF FALSE (T is higher than Single, e.g., double):
        std::variant<std::vector<HalfPrecType>, std::vector<SinglePrecType>, std::vector<T>> // 3 types
    >;

    StorageVariant _data;
    double _ratio;

public:
    Defaultcompressor() {_ratio = 1;};

    void compress(T* data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    void decompress(T* dest) override;

    std::vector<T> decompressCopy() override;

    void decompressCopy(T* dest) override;

    double get_ratio() override;

    Defaultcompressor* copy() override {
      Defaultcompressor* newComp = new Defaultcompressor();
      newComp->_data = _data;
      return newComp;
    }

    
};

}



