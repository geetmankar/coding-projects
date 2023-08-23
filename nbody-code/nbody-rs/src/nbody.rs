use color_eyre::eyre::Result;
use kdam::{term::Colorizer, tqdm, Colour};
use ndarray::{prelude::*, ShapeError};
use ndarray::{s, stack, Array1, Array2, ArrayBase, Axis, Ix2, OwnedRepr};
use std::io::{stderr, IsTerminal};

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

#[doc = r"
Macro to calculate difference
between a Nx1 2D array and its transpose
(returns a 2D square matrix)
---------------------------
Arguments:
x : Nx1 array to calculate the diff with
---------------------------
Returns:
(x - transpose_of_x) : 2D square matrix
"]
macro_rules! tdiff {
    // transpose difference
    ($x:expr) => {{
        let shape = ($x.len(), $x.len());
        let defarr = Array::zeros(shape);
        let fi = $x
            .insert_axis(Axis(1))
            .broadcast(shape)
            .unwrap_or(defarr.view())
            .to_owned();
        fi.t().to_owned() - fi.view()
    }};
}

#[doc = r"
Macro to calculate product
of a Nx1 2D array and its transpose
(returns a 2D square matrix)
---------------------------
Arguments:
x : Nx1 array to calculate the product of
---------------------------
Returns:
(x * transpose_of_x) : 2D square matrix
"]
macro_rules! sqmatrixify_sq {
    ($x:expr) => {{
        let shape = ($x.len(), $x.len());
        let defarr = Array::default(shape);
        let fi = $x
            .insert_axis(Axis(1))
            .broadcast(shape)
            .unwrap_or(defarr.view())
            .to_owned();
        fi.clone() * fi.t()
    }};
}

#[doc = r"Calculating the acceleration on each particle
---------------------------
Arguments:
nbsys: NBodySystem
---------------------------
Returns:
accel: Array2<f64>, acceleration of each particle
"]
pub fn get_accel(nbsys: &NBodySystem) -> Result<Array2<f64>, ShapeError> {
    let pos = nbsys.pos.clone();
    let mass = nbsys.mass.clone();
    let g = nbsys.g;
    let soft: f64 = nbsys.soft;

    let (x, y, z) = (pos.column(0), pos.column(1), pos.column(2));
    let (dx, dy, dz) = (tdiff!(x), tdiff!(y), tdiff!(z));

    let r3 = dx.clone().to_owned().mapv(|x| x.powi(2))
        + dy.clone().to_owned().mapv(|x| x.powi(2))
        + dz.clone().to_owned().mapv(|x| x.powi(2))
        + soft.powi(2);

    let inv_r3 = r3.mapv(|x| match x > 0. {
        true => x.powf(-1.5),
        _ => 0.,
    });

    let ax = g * (dx * inv_r3.view()).dot(&mass);
    let ay = g * (dy * inv_r3.view()).dot(&mass);
    let az = g * (dz * inv_r3.view()).dot(&mass);

    let accel = stack(Axis(1), &[ax.view(), ay.view(), az.view()])?;

    return Ok(accel);
}

#[doc = r"Get K.E. and P.E. of the simulation

Arguments: NBodySystem
Returns:
KE: Kinetic Energy of the System
PE: Potential Energy of the System"]
pub fn get_energy(nbsys: &NBodySystem) -> (f64, f64) {
    let (mass, vel) = (nbsys.mass.clone(), nbsys.vel.clone());
    let shape = (vel.view().nrows(), vel.view().ncols());

    let ke = 0.5
        * (mass
            .insert_axis(Axis(1))
            .broadcast(shape)
            .unwrap_or(Array::zeros(shape).view())
            .to_owned()
            * vel.map(|x| x.powi(2)))
        .sum();
    let pos = nbsys.pos.clone();

    let g: f64 = nbsys.g;
    let soft: f64 = nbsys.soft;

    let (x, y, z) = (pos.column(0), pos.column(1), pos.column(2));

    let (dx, dy, dz) = (tdiff!(x), tdiff!(y), tdiff!(z));

    let r =
        (dx.mapv(|x| x.powi(2)) + dy.mapv(|x| x.powi(2)) + dz.mapv(|x| x.powi(2)) + soft.powi(2))
            .mapv(|x| x.powf(0.5));

    let inv_r = r.mapv(|x| match x > 0. {
        true => x.powf(-1.0),
        _ => 0.,
    });

    let pe = {
        let penergy = -g
            * (sqmatrixify_sq!(nbsys.mass.clone()) * inv_r.view())
                .triu(1)
                .sum();
        penergy
    };

    return (ke, pe);
}

#[doc = r"
Trait to implement getting the upper/lower triangular part 
of a 2D square matrix, by zeroing all the other elements
"]
pub trait Tri {
    fn triu(&self, k: usize) -> Self;
    fn tril(&self, k: usize) -> Self;
}

impl Tri for ArrayBase<OwnedRepr<f64>, Ix2> {
    fn triu(&self, k: usize) -> Self {
        let cols = self.len_of(Axis(1));

        let mut result_arr = Array2::<f64>::zeros((cols, cols).f());

        for (i, row) in self.axis_iter(Axis(0)).enumerate() {
            let slice_main = (i + k)..self.nrows();
            row.slice(s![slice_main.clone()]) //.to_owned()
                .assign_to(result_arr.slice_mut(s![i, slice_main.clone()]));
        }

        return result_arr;
    }

    fn tril(&self, k: usize) -> Self {
        return self.t().to_owned().triu(k).t().to_owned();
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
) -> Result<(Array3<f64>, Array1<f64>, Array1<f64>, Array1<f64>), ShapeError> {
    kdam::term::init(stderr().is_terminal());

    // set up arrays to store data
    let n = nbsys.mass.len();
    let n_iter = (t_end / dt).ceil() as usize;
    let mut pos_save = Array3::zeros((n, 3, n_iter + 1));
    let mut ke_save = Array1::zeros(n_iter + 1);
    let mut pe_save = Array1::zeros(n_iter + 1);
    let mut time = Array1::zeros(n_iter + 1);

    // get initial acceleration and energies
    nbsys.accel.assign(&get_accel(nbsys)?);

    (ke_save[0], pe_save[0]) = get_energy(nbsys);

    // save initial conditions
    pos_save.slice_mut(s![.., .., 0]).assign(&nbsys.pos);

    // initialize time
    let mut t = 0.0;

    for i in tqdm!(
        0..n_iter,
        desc = "Simulating...",
        animation = "fillup",
        bar_format = format!(
            "{}|{{animation}}| {}/s]",
            "{desc} {percentage:3.0}%".colorize("#38F5F2"),
            "{count}/{total} [{elapsed}<{remaining}, {rate:.2}{unit}".colorize("#38F5F2")
        ),
        colour = Colour::gradient(&["#38F554", "#38F5F2"]),
        unit = " steps"
    ) {
        // update positions
        nbsys.pos.assign(
            &(nbsys.pos.clone()
                + ((nbsys.vel.clone() * dt) + (0.5 * nbsys.accel.clone() * dt.powi(2))).view()),
        );
        // find new acceleration
        let accel_new = get_accel(nbsys)?;

        // update velocities
        nbsys
            .vel
            .assign(&(nbsys.vel.clone() + (0.5 * (nbsys.accel.clone() + accel_new.view()) * dt)));

        pos_save.slice_mut(s![.., .., i + 1]).assign(&nbsys.pos);
        (ke_save[i + 1], pe_save[i + 1]) = get_energy(nbsys);
        time[i + 1] = t;

        // update acceleration
        nbsys.accel.assign(&accel_new);
        // update time
        t += dt;
    }

    // return saved data
    return Ok((pos_save, ke_save, pe_save, time));
}
