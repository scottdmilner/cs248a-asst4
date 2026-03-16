module output;


public struct Pixel<T : IArithmetic, let N : uint>
{
    public T[N] pixels;

    public __init() {
        clear();
    }
    
    [mutating]
    public void clear() {
        for (uint i = 0; i < N; ++i) {
            pixels[i] = T(0);
        }
    }

    public __subscript(uint l) -> T {
        get { return pixels[l]; }
        set { pixels[l] = newValue; }
    }

    [mutating]
    public void increment(Pixel<T,N> a) {
        for (uint i = 0; i < N; ++i) {
            pixels[i] = pixels[i] + a[i];
        }
    }
}

public Pixel<T,N> operator+<T :IArithmetic, let N : uint>(Pixel<T,N> a, Pixel<T,N> b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] + b[i];
    }
    return retval;
}

public Pixel<T,N> operator+<T :IArithmetic, let N : uint>(Pixel<T,N> a, uint b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] + T(b);
    }
    return retval;
}

public Pixel<T,N> operator*<T :IArithmetic, let N : uint>(Pixel<T,N> a, uint b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] * T(b);
    }
    return retval;
}

public Pixel<T,N> operator/<T :IArithmetic, let N : uint>(Pixel<T,N> a, uint b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] / T(b);
    }
    return retval;
}