import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { FolderOpen, X, RefreshCw, Upload, Clock, CheckCircle, XCircle } from "lucide-react";
import { api } from "@/lib/api";

interface BatchJob {
  batch_id: string;
  status: string;
  folder_path: string;
  max_workers: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  total_images: number;
  processed_images: number;
  progress_percentage: number;
  error_message?: string;
}

const BulkUploadModule = () => {
  const [folderPath, setFolderPath] = useState("");
  const [uploading, setUploading] = useState(false);
  const [batches, setBatches] = useState<BatchJob[]>([]);
  const [currentBatch, setCurrentBatch] = useState<BatchJob | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchBatches = async () => {
    setLoading(true);
    try {
      const response = await api.getAllBatches();
      setBatches(response.batches);

      // Find currently running batch
      const running = response.batches.find((batch: BatchJob) =>
        batch.status === 'pending' || batch.status === 'processing'
      );
      setCurrentBatch(running || null);
    } catch (error) {
      console.error('Error fetching batches:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!folderPath) return;

    setUploading(true);

    try {
      const response = await api.uploadFolder(folderPath, 4);
      setFolderPath("");
      await fetchBatches(); // Refresh data
    } catch (error) {
      console.error('Error uploading folder:', error);
    } finally {
      setUploading(false);
    }
  };



  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'processing': return <Upload className="h-4 w-4 text-blue-500" />;
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  useEffect(() => {
    fetchBatches();
  }, []);

  useEffect(() => {
    if (!currentBatch || (currentBatch.status !== 'pending' && currentBatch.status !== 'processing')) {
      return;
    }

    const interval = setInterval(() => {
      fetchBatches();
    }, 10000);

    return () => clearInterval(interval);
  }, [currentBatch?.batch_id, currentBatch?.status]);

  return (
    <div className="animate-fade-in space-y-6">
      {/* Bulk Upload Section */}
      <div className="bg-card rounded-2xl p-6 shadow-lg border border-border">
        <h2 className="text-xl font-semibold text-foreground mb-4">Upload Bulk Images</h2>

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-muted-foreground">Select Folder</label>
            <div className="relative">
              <label className="flex items-center justify-center h-12 border-2 border-dashed border-border rounded-lg bg-muted/30 hover:bg-muted/50 cursor-pointer transition-all duration-300 hover:scale-105 group">
                <FolderOpen className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors mr-2" />
                <span className="text-sm text-muted-foreground group-hover:text-primary transition-colors">
                  {folderPath || "Click to select folder"}
                </span>
                <input
                  type="file"
                  {...({ webkitdirectory: "" } as any)}
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const files = e.target.files;
                    if (files && files.length > 0) {
                      const path = files[0].webkitRelativePath.split('/')[0];
                      setFolderPath(path);
                    }
                  }}
                  disabled={uploading}
                />
              </label>
              {folderPath && (
                <button
                  onClick={() => setFolderPath("")}
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
            disabled={!folderPath || uploading}
            className="w-full h-12 bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-all duration-300 hover:scale-105 shadow-md"
          >
            {uploading ? "Starting Upload..." : "Upload"}
          </Button>
        </div>
      </div>

      {/* Current Batch Dashboard */}
      {currentBatch && (
        <div className="bg-card rounded-2xl p-6 shadow-lg border border-border">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-foreground">Currently Processing</h2>
            <Button
              onClick={fetchBatches}
              disabled={loading}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>

          <div className="bg-muted/30 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              {getStatusIcon(currentBatch.status)}
              <span className="font-medium">{currentBatch.batch_id}</span>
              <span className="text-sm text-muted-foreground capitalize">({currentBatch.status})</span>
            </div>

            <p className="text-sm text-muted-foreground mb-3">{currentBatch.folder_path}</p>

            {currentBatch.total_images > 0 && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress: {currentBatch.processed_images}/{currentBatch.total_images}</span>
                  <span>{currentBatch.progress_percentage}%</span>
                </div>
                <Progress value={currentBatch.progress_percentage} className="w-full" />
              </div>
            )}

            {currentBatch.error_message && (
              <p className="text-destructive text-sm mt-2">{currentBatch.error_message}</p>
            )}
          </div>
        </div>
      )}

      {/* All Batches Table */}
      <div className="bg-card rounded-2xl p-6 shadow-lg border border-border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-foreground">All Batches</h2>
          <Button
            onClick={fetchBatches}
            disabled={loading}
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-3 text-sm font-medium text-muted-foreground">Batch ID</th>
                <th className="text-left py-2 px-3 text-sm font-medium text-muted-foreground">Status</th>
                <th className="text-left py-2 px-3 text-sm font-medium text-muted-foreground">Folder</th>
                <th className="text-left py-2 px-3 text-sm font-medium text-muted-foreground">Progress</th>
                <th className="text-left py-2 px-3 text-sm font-medium text-muted-foreground">Created</th>
              </tr>
            </thead>
            <tbody>
              {batches.map((batch) => (
                <tr key={batch.batch_id} className="border-b border-border/50">
                  <td className="py-3 px-3 text-sm font-mono">{batch.batch_id}</td>
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(batch.status)}
                      <span className="text-sm capitalize">{batch.status}</span>
                    </div>
                  </td>
                  <td className="py-3 px-3 text-sm text-muted-foreground">{batch.folder_path}</td>
                  <td className="py-3 px-3 text-sm">
                    {batch.total_images > 0 ? (
                      `${batch.processed_images}/${batch.total_images} (${batch.progress_percentage}%)`
                    ) : (
                      'N/A'
                    )}
                  </td>
                  <td className="py-3 px-3 text-sm text-muted-foreground">{formatDate(batch.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {batches.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              No batches found
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BulkUploadModule;
