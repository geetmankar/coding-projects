#![allow(unused_imports)]
mod nbody;
mod plotting;
use std::{env, fs::create_dir, path::Path, process::Command};

use clap::Parser;
use color_eyre::eyre::{Error, Result};
use colored::Colorize;

use nbody::{run_sim, NBodySystem};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use plotting::plot_nbodysystem;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgParser {
    #[arg(short, long, default_value = "false")]
    video: bool,
    #[arg(short, long, default_value = "false")]
    images: bool,
}

fn main() -> Result<(), Error> {
    env::set_var("RUST_BACKTRACE", "1");

    let args = CliArgParser::parse();
    let n = 100; // Number of particles
    let t_end = 10.; // Time at which the sim ends
    let dt = 0.01; // 0.01 Timestep
    let soft = 0.1; // softening length
    let g = 3.; // Newton's gravitational constant

    let mass = Array1::<f64>::from_elem(n, 20.) * (n as f64).powi(-1); // total mass of the N particles
    let pos = Array2::<f64>::random((n, 3), Uniform::new(0., 3.0)); // random positions
    let vel = Array2::<f64>::random((n, 3), Uniform::new(0., 0.5));
    // let pos = Array2::<f64>::random((n, 3), Normal::new(0., 1.0)?); // random positions
    // let vel = Array2::<f64>::random((n, 3), Normal::new(0., 0.5)?);
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

    let filename = format!("./{dir}/nbsys").to_string();

    plot_nbodysystem(pos, ke, pe, t_all, filename)?;

    println!("{}", "Making video from images...".bold().bright_cyan());

    if args.video {
        Command::new("chmod").args(["+x", "mkvideo.sh"]).status()?;
        Command::new("./mkvideo.sh").status()?;

        println!("{}", "Video saved!".bold().bright_cyan());

        if !args.images {
            println!("{}", "Deleting images...".bold().bright_red());
            Command::new("rm").args(["-rf", "images"]).status()?;
            println!("{}", "Deleted".bold().on_bright_magenta());
        }
    }

    Ok(())
}
