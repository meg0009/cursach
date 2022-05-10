use crate::local::LocalPolygon;
use crate::mesh::Mesh;
use crate::triangle::{IndexedPolygon, IndexedTriangle, Polygon, Triangle};
use crate::ENABLE_JOIN;
use nalgebra as na;

const DELIMITER: f64 = 4.;

pub enum InOrOut {
    //находится в зоне 1
    In,
    //находится в зоне 2
    Out(Points, Points, na::Vector3<f64>),
}

#[derive(PartialEq, Eq)]
pub enum Points {
    X,
    Y,
    Z,
}

impl From<(&na::Vector3<f64>, &LocalPolygon)> for InOrOut {
    fn from((x0, local): (&na::Vector3<f64>, &LocalPolygon)) -> Self {
        if ENABLE_JOIN {
            let poly = local.get_poly();
            let x = poly.triangle().x().clone();
            let y = poly.triangle().y().clone();
            let z = poly.triangle().z().clone();
            let xy = y - x;
            let xz = z - x;
            let x_angle = xy.angle(&xz);
            let x_delimiter = x_angle / DELIMITER;
            let yx = x - y;
            let yz = z - y;
            let y_angle = yx.angle(&yz);
            let y_delimiter = y_angle / DELIMITER;
            let zx = x - z;
            let zy = y - z;
            let z_angle = zx.angle(&zy);
            let z_delimiter = z_angle / DELIMITER;

            //вращение против часовой стрелки
            let axisangle_anticlockwise = na::Vector3::z() * (-x_delimiter);
            let rot_x = na::Rotation3::new(axisangle_anticlockwise);
            let xa = rot_x * xy;

            let axisangle_anticlockwise = na::Vector3::z() * (-y_delimiter);
            let rot_y = na::Rotation3::new(axisangle_anticlockwise);
            let yc = rot_y * yz;

            let axisangle_anticlockwise = na::Vector3::z() * (-z_delimiter);
            let rot_z = na::Rotation3::new(axisangle_anticlockwise);
            let zb = rot_z * zx;

            // вращение по часовой стрелке
            let axisangle_clockwise = na::Vector3::z() * x_delimiter;
            let rot_x = na::Rotation3::new(axisangle_clockwise);
            let xb = rot_x * xz;

            let axisangle_clockwise = na::Vector3::z() * y_delimiter;
            let rot_y = na::Rotation3::new(axisangle_clockwise);
            let ya = rot_y * yx;

            let axisangle_clockwise = na::Vector3::z() * z_delimiter;
            let rot_z = na::Rotation3::new(axisangle_clockwise);
            let zc = rot_z * zy;

            let a = found_cross(&x, &xa, &y, &ya);
            let b = found_cross(&x, &xb, &z, &zb);
            let c = found_cross(&y, &yc, &z, &zc);

            //в xya
            if local.inside_triangle(x, y, a, &x0) {
                return InOrOut::Out(Points::X, Points::Y, a);
            }

            //в xzb
            let axisangle = na::Vector3::z() * (x_angle - x_delimiter);
            let rot_x = na::Rotation3::new(axisangle);
            if local.inside_triangle(x, rot_x * b, rot_x * z, &(rot_x * x0)) {
                return InOrOut::Out(Points::X, Points::Z, b);
            }

            //в yzc
            let axisangle = na::Vector3::z() * (y_delimiter - y_angle);
            let rot_y = na::Rotation3::new(axisangle);
            if local.inside_triangle(
                rot_y * (c - y),
                rot_y * (z - y),
                na::Vector3::zeros(),
                &(rot_y * (x0 - y)),
            ) {
                return InOrOut::Out(Points::Y, Points::Z, c);
            }

            InOrOut::In
        } else {
            InOrOut::In
        }
    }
}

fn found_cross(
    x1: &na::Vector3<f64>,
    x2: &na::Vector3<f64>,
    y1: &na::Vector3<f64>,
    y2: &na::Vector3<f64>,
) -> na::Vector3<f64> {
    let ay = (x2[1] * (x1[0] - x2[0]) / (x1[1] - x2[1])
        - y2[1] * (y1[0] - y2[0]) / (y1[1] - y2[1])
        + y2[0]
        - x2[0])
        / ((x1[0] - x2[0]) / (x1[1] - x2[1]) - (y1[0] - y2[0]) / (y1[1] - y2[1]));
    let ax = (ay - x2[1]) / (x1[1] - x2[1]) * (x1[0] - x2[0]) + x2[0];
    na::Vector3::new(ax, ay, 0.)
}

pub fn get_inside_point(
    poly: &IndexedPolygon,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> na::Vector3<f64> {
    let local = LocalPolygon::from(&Polygon::from_indexed_polygon(poly, mesh.vertexes()));

    let triangle = local.get_poly().triangle();
    let x = triangle.x();
    let y = triangle.y();
    let z = triangle.z();

    let xy = y - x;
    let xz = z - x;
    let x_angle = xy.angle(&xz);
    let x_delimiter = x_angle / DELIMITER;

    let yx = x - y;
    let yz = z - y;
    let y_angle = yx.angle(&yz);
    let y_delimiter = y_angle / DELIMITER;

    let y_delimiter = y_angle / DELIMITER;
    let zx = x - z;
    let zy = y - z;
    let z_angle = zx.angle(&zy);
    let z_delimiter = z_angle / DELIMITER;

    let triangle = poly.triangle();

    match (triangle.x_index(), triangle.y_index(), triangle.z_index()) {
        (xi, yi, _) if xi == i && yi == j || xi == j && yi == i => {
            let axisangle_anticlockwise = na::Vector3::z() * (-x_delimiter);
            let rot_x = na::Rotation3::new(axisangle_anticlockwise);
            let xa = rot_x * xy;
            let axisangle_clockwise = na::Vector3::z() * y_delimiter;
            let rot_y = na::Rotation3::new(axisangle_clockwise);
            let ya = rot_y * yx;
            found_cross(&x, &xa, &y, &ya)
        }
        (xi, _, zi) if xi == i && zi == j || xi == j && zi == i => {
            let axisangle_clockwise = na::Vector3::z() * x_delimiter;
            let rot_x = na::Rotation3::new(axisangle_clockwise);
            let xb = rot_x * xz;
            let axisangle_anticlockwise = na::Vector3::z() * (-z_delimiter);
            let rot_z = na::Rotation3::new(axisangle_anticlockwise);
            let zb = rot_z * zx;
            found_cross(&x, &xb, &z, &zb)
        }
        (_, yi, zi) if yi == i && zi == j || yi == j && zi == i => {
            let axisangle_anticlockwise = na::Vector3::z() * (-y_delimiter);
            let rot_y = na::Rotation3::new(axisangle_anticlockwise);
            let yc = rot_y * yz;
            let axisangle_clockwise = na::Vector3::z() * z_delimiter;
            let rot_z = na::Rotation3::new(axisangle_clockwise);
            let zc = rot_z * zy;
            found_cross(&y, &yc, &z, &zc)
        }
        _ => unreachable!(),
    }
}

pub fn get_mn(
    vec: &na::Vector3<f64>,
    a: &na::Vector3<f64>,
    poly_a: &IndexedPolygon,
    b: &na::Vector3<f64>,
    poly_b: &IndexedPolygon,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> (na::Vector3<f64>, na::Vector3<f64>) {
    (
        calculate_m(vec, a, poly_a, mesh, i, j),
        calculate_m(vec, b, poly_b, mesh, i, j),
    )
}

fn calculate_m(
    vec: &na::Vector3<f64>,
    a: &na::Vector3<f64>,
    poly_a: &IndexedPolygon,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> na::Vector3<f64> {
    let local = LocalPolygon::from(&Polygon::from_indexed_polygon(poly_a, mesh.vertexes()));

    let vec_local = local.to_local(vec);

    /*let triangle = local.get_poly().triangle();
    let x = triangle.x();
    let y = triangle.y();
    let z = triangle.z();

    let xy = y - x;
    let xz = z - x;
    let x_angle = xy.angle(&xz);
    let x_delimiter = x_angle / DELIMITER;

    let yx = x - y;
    let yz = z - y;
    let y_angle = yx.angle(&yz);
    let y_delimiter = y_angle / DELIMITER;

    let y_delimiter = y_angle / DELIMITER;
    let zx = x - z;
    let zy = y - z;
    let z_angle = zx.angle(&zy);
    let z_delimiter = z_angle / DELIMITER;*/

    let triangle = poly_a.triangle();

    match (triangle.x_index(), triangle.y_index(), triangle.z_index()) {
        (xi, yi, _) if xi == i && yi == j || xi == j && yi == i => {
            let p = triangle.x(mesh.vertexes());
            let q = triangle.y(mesh.vertexes());
            find_m(p, q, a, vec_local, &local)
        }
        (xi, _, zi) if xi == i && zi == j || xi == j && zi == i => {
            let q = triangle.x(mesh.vertexes());
            let p = triangle.z(mesh.vertexes());
            find_m(p, q, a, vec_local, &local)
        }
        (_, yi, zi) if yi == i && zi == j || yi == j && zi == i => {
            let p = triangle.y(mesh.vertexes());
            let q = triangle.z(mesh.vertexes());
            find_m(p, q, a, vec_local, &local)
        }
        _ => unreachable!(),
    }
}

fn find_m(
    p: na::Vector3<f64>,
    q: na::Vector3<f64>,
    a: &na::Vector3<f64>,
    vec_local: na::Vector3<f64>,
    local: &LocalPolygon,
) -> na::Vector3<f64> {
    let local_p = local.to_local(&p);
    let local_q = local.to_local(&q);
    let b_x = ((local_p[0] - local_q[0]) / (local_p[1] - local_q[1]) * vec_local[0] + vec_local[1]
        - local_q[1]
        + local_q[0] * (local_p[1] - local_q[1]) / (local_p[0] - local_q[0]))
        / ((local_p[1] - local_q[1]) / (local_p[0] - local_q[0])
            + (local_p[0] - local_q[0]) / (local_p[1] - local_q[1]));
    let b_y = (local_p[1] - local_q[1]) / (local_p[0] - local_q[0]) * b_x + local_q[1]
        - local_q[0] * (local_p[1] - local_q[1]) / (local_p[0] - local_q[0]);
    let b = na::Vector3::new(b_x, b_y, 0.);
    let mpx = (b_x * (vec_local[1] - b_y) / (vec_local[0] - b_x)
        - local_p[0] * (a[1] - local_p[1]) / (a[0] - local_p[0])
        + local_p[1])
        / ((vec_local[1] - b_y) / (vec_local[0] - b_x) - (a[1] - local_p[1]) / (a[0] - local_p[0]));
    let mpy = (mpx - b_x) / (vec_local[0] - b_x) * (vec_local[1] - b_y) + b_y;
    let mqx = (b_x * (vec_local[1] - b_y) / (vec_local[0] - b_x)
        - local_q[0] * (a[1] - local_q[1]) / (a[0] - local_q[0])
        + local_q[1])
        / ((vec_local[1] - b_y) / (vec_local[0] - b_x) - (a[1] - local_q[1]) / (a[0] - local_q[0]));
    let mqy = (mqx - b_x) / (vec_local[0] - b_x) * (vec_local[1] - b_y) + b_y;
    let mp = na::Vector3::new(mpx, mpy, 0.);
    let mq = na::Vector3::new(mqx, mqy, 0.);
    if (mp - b).magnitude() < (mq - b).magnitude() {
        mp
    } else {
        mq
    }
}
