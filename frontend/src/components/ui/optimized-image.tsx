import { useState, useRef, useEffect, useCallback } from 'react';

interface OptimizedImageProps {
  src: string;
  alt: string;
  className?: string;
  onLoad?: () => void;
  lazy?: boolean;
}

export const OptimizedImage = ({ src, alt, className = '', onLoad, lazy = false }: OptimizedImageProps) => {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const [inView, setInView] = useState(!lazy);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleLoad = useCallback(() => {
    setLoaded(true);
    onLoad?.();
  }, [onLoad]);

  const handleError = useCallback(() => {
    setError(true);
  }, []);

  useEffect(() => {
    if (!lazy || inView) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: '50px' }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [lazy, inView]);

  useEffect(() => {
    if (!inView) return;
    
    const img = imgRef.current;
    if (!img) return;

    // Reset states when src changes
    setLoaded(false);
    setError(false);

    if (img.complete && img.naturalWidth > 0) {
      handleLoad();
    } else {
      img.addEventListener('load', handleLoad);
      img.addEventListener('error', handleError);
    }

    return () => {
      img.removeEventListener('load', handleLoad);
      img.removeEventListener('error', handleError);
    };
  }, [src, inView, handleLoad, handleError]);

  return (
    <div ref={containerRef} className="relative w-full h-full">
      {!loaded && !error && (
        <div className="absolute inset-0 bg-gray-200 animate-pulse rounded-2xl" />
      )}
      {error && (
        <div className="absolute inset-0 bg-gray-100 flex items-center justify-center rounded-2xl">
          <span className="text-gray-400 text-sm">Failed to load</span>
        </div>
      )}
      {inView && (
        <img
          ref={imgRef}
          src={src}
          alt={alt}
          className={`${className} ${loaded ? 'opacity-100' : 'opacity-0'} transition-opacity duration-150`}
          loading={lazy ? "lazy" : "eager"}
          decoding="async"
          fetchpriority={lazy ? "low" : "high"}
        />
      )}
    </div>
  );
};