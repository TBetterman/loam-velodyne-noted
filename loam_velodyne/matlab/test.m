clear all;
clc;
syms lx;
syms ly;
syms lz;
syms cx;
syms cy;
syms cz;
[Rlx,Rly,Rlz] = rotationMatrix(lx,ly,lz);
[Rcx,Rcy,Rcz] = rotationMatrix(cx,cy,cz);
rotationl = Rlz*Rlx*Rly;
rotationc = Rcz*Rcx*Rcy;
rotationFinall = ((rotationl))*rotationc