{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python310;
  pythonPackages = python.pkgs;

in pkgs.mkShell {
  buildInputs = [
    python
    pythonPackages.pip
    pythonPackages.virtualenv
    pythonPackages.numpy
    pythonPackages.cmake
    pkgs.gdb    # Add gdb here
    pkgs.cargo
    pkgs.rustc
  ];

  shellHook = ''
    # Create and activate a virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
      virtualenv venv
    fi
    source venv/bin/activate
  '';
}
