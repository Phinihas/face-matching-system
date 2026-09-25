const Navbar = () => {
  return (
    <nav className="h-16 bg-card border-b border-border flex items-center justify-between px-8 shadow-sm">
      <div className="flex items-center gap-4 ">
        <img src="logo.png" alt="Face Match AI Logo" className="h-10 w-10" />
        <h1 className="text-xl font-semibold text-foreground">
          FaceMatch
        </h1>
      </div>
    </nav>
  );
};

export default Navbar;
