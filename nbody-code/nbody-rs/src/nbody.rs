#![allow(unused_imports, dead_code, unused_variables)]

// use nalgebra::Matrix2;
use kdam::tqdm;
use ndarray::prelude::*;
use ndarray::{
    s, stack, ArcArray2, Array, Array1, Array2, Array3, ArrayBase, Axis, Dim, Dimension, Ix2,
    OwnedRepr, RawData, ViewRepr,
};

#[doc = r"a struct of n massive bodies with
positions, masses, velocities, accelerations,
gravitaional constant g, and the softening length
stored in it (length at which to stop calculating the force
and replacing it with a small value, since Newton's
gravity becomes infinite as distance between 2 point
particles tends to zero)"]
#[derive(Debug)]
pub struct NBodySystem {
    pub n: usize,
    pub g: f64,
    pub soft: f64,
    pub pos: Array2<f64>,   // N x 3
    pub mass: Array1<f64>,  // N x 1
    pub vel: Array2<f64>,   // N x 3
    pub accel: Array2<f64>, // N x 3
}

#[doc = r"Calculating the acceleration on each particle
---------------------------
Arguments:
---------------------------
nbsys: NBodySystem
---------------------------
Returns:
---------------------------
accel: Array2<f64>, acceleration of each particle
"]
pub fn get_accel(nbsys: &NBodySystem) -> Array2<f64> {
    let pos = nbsys.pos.clone();
    let mass = nbsys.mass.clone();
    let g = nbsys.g.clone();
    let soft: f64 = nbsys.soft.clone();

    let shape = (pos.view().len(), pos.view().len());
    let mut r3 = Array2::<f64>::zeros(shape.f());

    let (x, y, z) = (pos.column(0), pos.column(1), pos.column(2));
    let (dx, dy, dz) = (
        x.t().broadcast(shape).unwrap().to_owned() - x.view().broadcast(shape).unwrap_or(r3.view()),
        y.t().broadcast(shape).unwrap().to_owned() - y.view().broadcast(shape).unwrap_or(r3.view()),
        z.t().broadcast(shape).unwrap().to_owned() - z.view().broadcast(shape).unwrap_or(r3.view()),
    );

    r3 = dx.clone().to_owned() * dx.clone()
        + dy.clone().to_owned() * dy.clone()
        + dz.clone().to_owned() * dz.clone()
        + soft.powi(2);

    let inv_r3 = r3.mapv(|x| match x > 0. {
        true => x.powf(-1.5),
        _ => 0.,
    });

    let ax = (g * (dx.clone().to_owned() * inv_r3.clone())).dot(&mass);
    let ay = (g * (dy.clone().to_owned() * inv_r3.clone())).dot(&mass);
    let az = (g * (dz.clone().to_owned() * inv_r3.clone())).dot(&mass);

    let accel = stack(Axis(0), &[ax.view(), ay.view(), az.view()]).unwrap();

    return accel;
}

#[doc = r"Get K.E. and P.E. of the simulation

Arguments: NBodySystem
Returns:
KE: Kinetic Energy of the System
PE: Potential Energy of the System"]
pub fn get_energy<'a>(nbsys: &'a NBodySystem) -> (f64, f64) {
    let ke = 0.5 * (nbsys.mass.clone() * nbsys.vel.clone().mapv(|x| x.powi(2))).sum();

    let pos = nbsys.pos.clone();
    let g: f64 = nbsys.g.clone();
    let soft: f64 = nbsys.soft.clone();
    let shape = (pos.view().len(), pos.view().len());
    let mut r3 = Array2::<f64>::zeros(shape.f());

    let (x, y, z) = (pos.column(0), pos.column(1), pos.column(2));
    let (dx, dy, dz) = (
        x.t().broadcast(shape).unwrap().to_owned() - x.view().broadcast(shape).unwrap_or(r3.view()),
        y.t().broadcast(shape).unwrap().to_owned() - y.view().broadcast(shape).unwrap_or(r3.view()),
        z.t().broadcast(shape).unwrap().to_owned() - z.view().broadcast(shape).unwrap_or(r3.view()),
    );

    r3 = dx.clone().to_owned() * dx.clone()
        + dy.clone().to_owned() * dy.clone()
        + dz.clone().to_owned() * dz.clone()
        + soft.powi(2);

    let inv_r3 = r3.mapv(|x| match x > 0. {
        true => x.powi(-1),
        _ => 0.,
    });

    let pe = {
        let intermediate = nbsys.g
            * nbsys
                .mass
                .view()
                .broadcast((nbsys.mass.len_of(Axis(0)), nbsys.mass.len_of(Axis(1))))
                .unwrap()
                .dot(&nbsys.mass.t())
                .mapv(|x| -x);

        let penergy = (intermediate * inv_r3.view()).triu(1).sum();
        penergy
    };

    return (ke, pe);
}

#[doc = r"implement getting the upper/lower triangular part 
of a 2D square matrix, by zeroing all the other elements"]
pub trait Tri {
    fn triu(&self, k: usize) -> Self;
    fn tril(&self, k: usize) -> Self;
}

impl Tri for ArrayBase<OwnedRepr<f64>, Ix2> {
    fn triu(&self, k: usize) -> Self {
        let cols = self.len_of(Axis(1));
        let copied_arr = self.clone();

        let mut result_arr = Array2::<f64>::zeros((cols, cols).f());

        for (i, row) in copied_arr.axis_iter(Axis(0)).enumerate() {
            let slice_main = (i as usize + k)..self.len_of(Axis(0));
            row.to_owned()
                .slice(s![slice_main.clone()])
                .assign_to(result_arr.slice_mut(s![i, slice_main.clone()]));
        }

        return result_arr;
    }

    fn tril(&self, k: usize) -> Self {
        return self.clone().t().to_owned().triu(k).t().to_owned();
    }
}

#[doc = r"
Runs the simulation for a specified duration.

Arguments:
nbsys : N-Body System
t_end : Duration of simulation
dt    : time step for simulation

Returns:
pos_save: Particle positions across the simulation
ke_save : K.E. of the system accross the simulation
pe_save : P.E. of the system accross the simulation
time    : time-steps of the simulation
"]
pub fn run_sim(
    nbsys: &mut NBodySystem,
    t_end: f64,
    dt: f64,
) -> (Array3<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    // set up arrays to store data
    let n_iter = (t_end / dt).ceil() as usize;
    let mut pos_save = Array::zeros((nbsys.n, 3, n_iter + 1));
    let mut ke_save = Array::zeros(n_iter + 1);
    let mut pe_save = Array::zeros(n_iter + 1);
    let mut time = Array::zeros(n_iter + 1);

    // get initial acceleration and energies
    nbsys.accel = get_accel(&nbsys);
    (ke_save[0], pe_save[0]) = get_energy(&nbsys);

    // save initial conditions
    pos_save.slice_mut(s![.., .., 0]).assign(&nbsys.pos);

    // initialize time
    let mut t = 0.0;

    for i in tqdm!(0..n_iter) {
        // update positions
        nbsys.pos = &nbsys.pos + &(&nbsys.vel * dt) + &(0.5 * &nbsys.accel * dt.powi(2));

        // find new acceleration
        let accel_new = get_accel(&nbsys);

        // update velocities
        nbsys.vel = &nbsys.vel + &(0.5 * (&nbsys.accel + &accel_new) * dt);

        pos_save.slice_mut(s![.., .., i + 1]).assign(&nbsys.pos);
        let (ke, pe) = get_energy(&nbsys);
        ke_save[i + 1] = ke;
        pe_save[i + 1] = pe;
        time[i + 1] = t;

        // update acceleration
        nbsys.accel = accel_new;
        // update time
        t += dt;
    }

    // return saved data
    (pos_save, ke_save, pe_save, time)
}
