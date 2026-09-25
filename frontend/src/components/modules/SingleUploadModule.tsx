import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Image, X } from "lucide-react";
import { api } from "@/lib/api";

const SingleUploadModule = () => {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    
    setUploading(true);
    setResult(null);
    
    try {
      const response = await api.uploadSingleImage(selectedFile);
      setResult({ success: true, ...response });
      setSelectedFile(null);
    } catch (error) {
      console.error('Error uploading single image:', error);
      setResult({ error: 'Failed to upload image' });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="bg-card rounded-2xl p-8 shadow-lg border border-border">
        <h2 className="text-2xl font-semibold text-foreground mb-2">Upload Single Image</h2>
        <p className="text-muted-foreground mb-6">Select and upload a single image to process and add to the database.</p>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-muted-foreground">Select Image</label>
            <div className="relative">
              <label className="flex items-center justify-center h-32 border-2 border-dashed border-border rounded-lg bg-muted/30 hover:bg-muted/50 cursor-pointer transition-all duration-300 hover:scale-105 group">
                <div className="text-center">
                  <Image className="h-8 w-8 text-muted-foreground group-hover:text-primary transition-colors mx-auto mb-2" />
                  <span className="text-sm text-muted-foreground group-hover:text-primary transition-colors">
                    {selectedFile ? selectedFile.name : "Click to select image"}
                  </span>
                  <p className="text-xs text-muted-foreground mt-1">
                    Supports JPG, PNG, GIF, WebP
                  </p>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handleFileSelect(file);
                    }
                  }}
                  disabled={uploading}
                />
              </label>
              {selectedFile && (
                <button
                  onClick={() => setSelectedFile(null)}
                  className="absolute -top-2 -right-2 h-6 w-6 bg-destructive text-destructive-foreground rounded-full flex items-center justify-center hover:bg-destructive/80 transition-colors"
                  disabled={uploading}
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>
          </div>
          
          <Button
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
            className="w-full h-12 bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-all duration-300 hover:scale-105 shadow-md"
          >
            {uploading ? "Uploading..." : "Upload Image"}
          </Button>
          
          {result && (
            <div className="bg-muted/30 rounded-xl p-4 animate-fade-in">
              {result.error ? (
                <p className="text-destructive">{result.error}</p>
              ) : (
                <div>
                  <p className="text-primary font-semibold">Image uploaded successfully!</p>
                  <div className="text-sm text-muted-foreground mt-2">
                    <p>Request ID: {result.request_id}</p>
                    <p>Status: {result.status}</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SingleUploadModule;