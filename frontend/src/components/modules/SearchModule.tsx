import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Upload, X } from "lucide-react";
import { api } from "@/lib/api";
import { OptimizedImage } from "@/components/ui/optimized-image";


interface SearchResultItem {
  name: string;
  url: string;
  score: number;
}

const SearchModule = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [results, setResults] = useState<SearchResultItem[] | null>(null);
  const [timeTaken, setTimeTaken] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [noResultsMessage, setNoResultsMessage] = useState<string>("");
  const [topK, setTopK] = useState(3);

  const handleImageUpload = (file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedImage(reader.result as string);
    };
    reader.readAsDataURL(file);
    setUploadedFile(file);
    setResults(null);
  };



  const handleSearch = async () => {
    if (!uploadedFile) return;

    setLoading(true);
    setResults([]);
    setNoResultsMessage("");

    try {
      const apiResult = await api.searchFaces(uploadedFile, topK, (progressImages) => {
        // Update results progressively as images are processed
        setResults([...progressImages]);
        // Show UI immediately after first image
        setLoading(false);
      });
      
      setTimeTaken(apiResult.searchTime || '');
      
      if (apiResult.images.length === 0 && apiResult.message) {
        setNoResultsMessage(apiResult.message);
        setResults([]);
      }
      
      setLoading(false);
    } catch (error) {
      console.error('Error searching faces:', error);
      setLoading(false);
      setResults(null);
    }
  };

  return (
    <div className="h-[calc(100vh-6rem)] bg-gray-50 overflow-hidden p-6">
      {(!results || results.length === 0) && !noResultsMessage ? (
        <div className="h-full flex items-center justify-center">
          <div className="bg-white rounded-3xl shadow-xl p-8 w-full max-w-4xl">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Face Search</h2>
              <p className="text-sm text-gray-500">Upload an image to find similar faces</p>
            </div>

            <div className="flex gap-8 items-center">
              <div className="flex-1 group">
                <div className="relative">
                  <label className="block">
                    <div className="relative h-80 w-full border-2 border-dashed border-gray-300 rounded-2xl bg-gray-50 hover:bg-gray-100 cursor-pointer transition-all duration-300 hover:border-blue-400 hover:shadow-md group overflow-hidden">
                      {uploadedImage ? (
                        <img src={uploadedImage} alt="Query" className="w-full h-full object-contain rounded-2xl group-hover:blur-sm group-hover:opacity-70 transition-all duration-300" />
                      ) : (
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <Upload className="h-16 w-16 text-gray-400 group-hover:text-blue-500 transition-colors mb-4" />
                          <p className="text-base font-medium text-gray-600 group-hover:text-gray-900 transition-colors">
                            Click to upload image
                          </p>
                          <p className="text-xs text-gray-400 mt-1">PNG, JPG or JPEG</p>
                        </div>
                      )}
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleImageUpload(file);
                      }}
                    />
                  </label>
                  {uploadedImage && (
                    <button
                      onClick={() => {
                        setUploadedImage(null);
                        setUploadedFile(null);
                      }}
                      className="absolute top-3 right-3 bg-red-500 rounded-full p-2 shadow-lg opacity-0 group-hover:opacity-100 transition-all duration-300 z-10 hover:bg-red-600"
                    >
                      <X className="h-5 w-5 text-white" />
                    </button>
                  )}
                </div>
              </div>

              <div className="flex-shrink-0 w-80 space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Maximum Results
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all text-center text-lg font-medium"
                  />
                </div>

                <Button
                  onClick={handleSearch}
                  disabled={!uploadedImage || loading}
                  className="w-full h-14 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <div className="flex items-center gap-3">
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Searching...
                    </div>
                  ) : (
                    'Search Similar Faces'
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      ) : noResultsMessage ? (
        <div className="h-full flex items-center justify-center">
          <div className="bg-white rounded-3xl shadow-xl p-8 w-full max-w-2xl text-center">
            <div className="w-16 h-16 rounded-xl overflow-hidden shadow-md ring-2 ring-gray-200 mx-auto mb-6">
              <img src={uploadedImage!} alt="Query" className="w-full h-full object-cover" />
            </div>
            <div className="mb-6">
              <h3 className="text-xl font-bold text-gray-900 mb-2">No Matches Found</h3>
              <p className="text-gray-600">{noResultsMessage}</p>
              {timeTaken && (
                <p className="text-sm text-gray-500 mt-2">Search completed in {timeTaken}</p>
              )}
            </div>
            <Button
              onClick={() => {
                setResults(null);
                setNoResultsMessage("");
              }}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl px-8 py-3"
            >
              Try Another Search
            </Button>
          </div>
        </div>
      ) : (
        <div className="h-full animate-in fade-in duration-500">
          <div className="bg-white rounded-3xl shadow-xl h-full flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-white">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-xl overflow-hidden shadow-md ring-2 ring-gray-200">
                  <img src={uploadedImage!} alt="Query" className="w-full h-full object-cover" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900">Search Results</h3>
                  <p className="text-sm text-gray-500">
                    {timeTaken ? `Found in ${timeTaken}` : `${results.length} results found`}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-semibold shadow-sm">
                  {results.length} matches
                </div>
                <Button
                  onClick={() => {
                    setResults(null);
                    setNoResultsMessage("");
                  }}
                  variant="outline"
                  className="rounded-xl font-medium"
                >
                  New Search
                </Button>
              </div>
            </div>

            <div className="flex-1 p-6 overflow-auto">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-5">
                {results.map((result, idx) => (
                  <ResultCard key={result.name} result={result} index={idx} />
                ))}
                {loading && (
                  <div className="bg-white rounded-2xl shadow-md border border-gray-100 h-64 flex items-center justify-center">
                    <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const ResultCard = ({ result, index }: { result: SearchResultItem; index: number }) => (
  <div className="bg-white rounded-2xl shadow-md hover:shadow-lg transition-shadow duration-150 border border-gray-100">
    <div className="h-64 relative overflow-hidden rounded-2xl">
      <OptimizedImage
        src={result.url}
        alt={`Match ${index + 1}`}
        className="w-full h-full object-cover rounded-2xl"
        lazy={index >= 2}
      />
      <div className="absolute top-3 right-3">
        <div className="bg-white/95 px-3 py-1.5 rounded-full text-sm font-bold text-gray-900 shadow-lg">
          {Math.round(result.score * 100)}%
        </div>
      </div>
    </div>
  </div>
);

export default SearchModule;