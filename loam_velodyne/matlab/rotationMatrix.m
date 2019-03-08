function [Rx,Ry,Rz] = rotationMatrix(anglex,angley,anglez)
    Rx = [1 0 0;
          0 cos(anglex) -sin(anglex);
          0 sin(anglex) cos(anglex)];
      
    Ry = [cos(angley) 0 sin(angley);
          0           1 0;
          -sin(angley) 0 cos(angley)];
      
    Rz = [cos(anglez) -sin(anglez) 0;
          sin(anglez) cos(anglez) 0;
          0           0           1];
end