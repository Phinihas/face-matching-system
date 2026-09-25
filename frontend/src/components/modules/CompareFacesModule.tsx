import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";
import CircularProgress from "@/components/ui/circular-progress";
import { api, CompareResult } from "@/lib/api";

const CompareFacesModule = () => {
  const [image1, setImage1] = useState<string | null>(null);
  const [image2, setImage2] = useState<string | null>(null);
  const [image1File, setImage1File] = useState<File | null>(null);
  const [image2File, setImage2File] = useState<File | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (file: File, setImage: (url: string) => void, setFile: (file: File) => void) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setImage(reader.result as string);
    };
    reader.readAsDataURL(file);
    setFile(file);
  };

  const handleCompare = async () => {
    if (!image1File || !image2File) return;

    setLoading(true);
    try {
      const result = await api.compareFaces(image1File, image2File);
      setResult(result);
    } catch (error) {
      console.error('Error comparing faces:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="bg-card rounded-2xl p-8 shadow-lg border border-border">
        <h2 className="text-2xl font-semibold text-foreground mb-2">Compare Two Faces</h2>
        <p className="text-muted-foreground mb-6">Upload two images to compare facial similarity.</p>

        <div className="grid grid-cols-2 gap-6 mb-6">
          {[
            { image: image1, setImage: setImage1, setFile: setImage1File, label: "Image 1" },
            { image: image2, setImage: setImage2, setFile: setImage2File, label: "Image 2" }
          ].map((item, idx) => (
            <div key={idx} className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">{item.label}</label>
              <label className="flex flex-col items-center justify-center h-64 border-2 border-dashed border-border rounded-xl bg-muted/30 hover:bg-muted/50 cursor-pointer transition-all duration-300 hover:scale-105 group overflow-hidden">
                {item.image ? (
                  <img src={item.image} alt={item.label} className="h-full w-full object-contain rounded-xl" />
                ) : (
                  <div className="text-center">
                    <Upload className="mx-auto h-12 w-12 text-muted-foreground group-hover:text-primary transition-colors" />
                    <p className="mt-2 text-sm text-muted-foreground">Click to upload</p>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleImageUpload(file, item.setImage, item.setFile);
                  }}
                />
              </label>
            </div>
          ))}
        </div>

        <Button
          onClick={handleCompare}
          disabled={!image1 || !image2 || loading}
          className="w-full h-12 bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-all duration-300 hover:scale-105 shadow-md mb-6"
        >
          {loading ? 'Comparing...' : 'Compare'}
        </Button>

        {result && (
          <div className="bg-muted/30 rounded-xl p-6 text-center animate-scale-in">
            <CircularProgress value={result.similarity} size={140} />
            <p className="mt-4 text-2xl font-semibold">
              {result.match ? (
                <span className="text-primary">Match Found!</span>
              ) : (
                <span className="text-destructive">No Match</span>
              )}
            </p>
            <p className="text-muted-foreground mt-2">Similarity: {result.similarity}%</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CompareFacesModule;
