#include <iostream>

class Point {
public:
    int x, y;

    Point(int x, int y) : x(x), y(y) {}

    void print() {
        std::cout << "Point(" << x << ", " << y << ")" << std::endl;
    }

    void setX(int newX) {
        x = newX;
    }

    void setY(int newY) {
        y = newY;
    }

    int getX() {
        return x;
    }

    int getY() {
        return y;
    }
};

extern "C" Point* create_point(int x, int y) {
    return new Point(x, y);
}

extern "C" void delete_point(Point* p) {
    delete p;
}

extern "C" void print_point(Point* p) {
    p->print();
}

extern "C" void set_x(Point* p, int x) {
    p->setX(x);
}

extern "C" void set_y(Point* p, int y) {
    p->setY(y);
}

extern "C" int get_x(Point* p) {
    return p->getX();
}

extern "C" int get_y(Point* p) {
    return p->getY();
}
