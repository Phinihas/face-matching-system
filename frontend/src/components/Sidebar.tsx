import { Upload, Users, Search, Image } from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarProps {
  activeModule: string;
  onModuleChange: (module: string) => void;
}

const modules = [
  { id: "single", label: "Single Upload", icon: Image },
  { id: "bulk", label: "Bulk Upload", icon: Upload },
  { id: "compare", label: "Compare Faces", icon: Users },
  { id: "search", label: "1:N Search", icon: Search },
];

const Sidebar = ({ activeModule, onModuleChange }: SidebarProps) => {
  return (
    <aside className="w-64 bg-card border-r border-border flex flex-col p-4 gap-2">
      {modules.map((module) => {
        const Icon = module.icon;
        const isActive = activeModule === module.id;
        
        return (
          <button
            key={module.id}
            onClick={() => onModuleChange(module.id)}
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-300",
              "hover:scale-105 hover:shadow-md",
              isActive
                ? "bg-gradient-to-r from-primary to-secondary text-primary-foreground shadow-lg"
                : "bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground"
            )}
          >
            <Icon className={cn("h-5 w-5", isActive && "animate-pulse")} />
            <span className="font-medium">{module.label}</span>
          </button>
        );
      })}
    </aside>
  );
};

export default Sidebar;
