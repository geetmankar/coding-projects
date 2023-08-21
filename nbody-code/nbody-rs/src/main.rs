mod nbody;
mod plotting;
use std::fs::create_dir;
use std::path::Path;

use clap::Parser;
use color_eyre::eyre::{Error, Result};
use kdam::tqdm;
use nbody::{run_sim, NBodySystem};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use plotting::plot_nbodysystem;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgParser {
    #[arg(short, long, default_value = "false")]
    video: bool,
}

// fn main() -> Result<(), Error> {
//     let dir = "images";

//     let dirpath = Path::new(&dir);
//     if !dirpath.exists() {
//         create_dir(dirpath)?
//     }

//     println!("{dir}/nbsys.png");
//     Ok(())
// }

fn main() -> Result<(), Error> {
    let args = CliArgParser::parse();
    let n = 100; // Number of particles
    let t_end = 10.; // Time at which the sim ends
    let dt = 0.1; // 0.01 Timestep
    let soft = 0.1; // softening length
    let g = 3.; // Newton's gravitational constant

    let mass = Array1::<f64>::from_elem(n, 20.); // total mass of the N particles
    let pos = Array2::<f64>::random((n, 3), Normal::new(0., 2.).unwrap()); // random positions
    let vel = Array2::<f64>::random((n, 3), Normal::new(0., 1.).unwrap());
    // let pos = Array2::<f64>::random((n, 3), Uniform::new(-4., 4.)); // random positions
    // let vel = Array2::<f64>::random((n, 3), Uniform::new(-1., 1.));
    let accel = Array2::<f64>::zeros((n, 3));
    let mut nbsys = NBodySystem {
        n,
        g,
        soft,
        pos,
        mass,
        vel,
        accel,
    };

    let (pos, ke, pe, t_all) = run_sim(&mut nbsys, t_end, dt)?;

    let dir = "images";

    let dirpath = Path::new(&dir);
    if !dirpath.exists() {
        create_dir(dirpath)?
    }

    for i in tqdm!(
        0..t_all.len(),
        desc = "Plotting...",
        colour = "green",
        unit = " frames"
    ) {
        let filename = format!("./{dir}/nbsys_{i}.png").to_string();

        let ke_i = ke.clone().slice(s![..=i]).to_owned();
        let pe_i = pe.clone().slice(s![..=i]).to_owned();
        let t_all_i = t_all.clone().slice(s![..=i]).to_owned();

        plot_nbodysystem(pos.clone(), ke_i, pe_i, t_all_i, i, Some(filename.as_str()))?;
    }

    if args.video {
        todo!()
    }
    Ok(())
}
