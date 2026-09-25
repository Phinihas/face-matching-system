import { useState } from "react";
import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";
import SingleUploadModule from "@/components/modules/SingleUploadModule";
import BulkUploadModule from "@/components/modules/BulkUploadModule";
import CompareFacesModule from "@/components/modules/CompareFacesModule";
import SearchModule from "@/components/modules/SearchModule";

const Index = () => {
  const [activeModule, setActiveModule] = useState("search");

  const renderModule = () => {
    switch (activeModule) {
      case "single":
        return <SingleUploadModule />;
      case "bulk":
        return <BulkUploadModule />;
      case "compare":
        return <CompareFacesModule />;
      case "search":
        return <SearchModule />;
      default:
        return <SearchModule />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col w-full">
      <Navbar />
      <div className="flex flex-1 w-full">
        <Sidebar activeModule={activeModule} onModuleChange={setActiveModule} />
        <main className="flex-1 p-4">
          {renderModule()}
        </main>
      </div>
    </div>
  );
};

export default Index;
