// #![allow(dead_code, unused_imports, unused_variables)]
use color_eyre::eyre::{Error, Result};
use kdam::{par_tqdm, rayon::prelude::*, term::Colorizer, Colour};
use ndarray::{s, Array1, Array3, Axis};
use ndarray_stats::QuantileExt;
use plotpy::{Curve, Plot};
use std::io::{stderr, IsTerminal};

macro_rules! energy_range {
    ($ke: expr, $pe: expr) => {
        (
            $pe.to_owned().min()?.to_owned() + 1.0,
            $ke.to_owned().max()?.to_owned() + 1.0,
        )
    };
}

fn max<T: PartialEq + PartialOrd>(x: T, y: T) -> T {
    match x > y {
        true => x,
        _ => y,
    }
}

pub fn plot_nbodysystem(
    pos: Array3<f64>,
    ke: Array1<f64>,
    pe: Array1<f64>,
    t_all: Array1<f64>,
    filename: String,
) -> Result<(), Error> {
    let e_range = energy_range!(ke.view(), pe.view());

    let tmax = t_all[t_all.len() - 1];

    kdam::term::init(stderr().is_terminal());

    par_tqdm!(
        (0..t_all.len()).into_par_iter(),
        desc = "Plotting...",
        bar_format = format!(
            "{}|{{animation}}| {}/s]",
            "{desc} {percentage:3.0}%".colorize("#EE6FF8"),
            "{count}/{total} [{elapsed}<{remaining}, {rate:.2}{unit}".colorize("#EE6FF8")
        ),
        colour = Colour::gradient(&["#5A56E0", "#EE6FF8"]),
        unit = " frames"
    )
    .for_each(|i| {
        let mut plot = Plot::new();
        plot.set_figure_size_inches(4.0, 6.0).set_gaps(0.2, 0.2);

        let fname = filename.clone() + &format!("_{i:04}.png");

        let pos_interim = pos.slice(s![.., .., i]).clone().to_owned();
        let pos_x = pos_interim.slice(s![.., 0]).to_vec();
        let pos_y = pos_interim.slice(s![.., 1]).to_vec();

        let trail_start = max(0i32, i as i32 - 10i32) as usize;

        let pos_trail_interim = pos.slice(s![.., .., trail_start..i]).clone().to_owned();

        let pos_trailx: Vec<Vec<f64>> = pos_trail_interim
            .slice(s![.., 0, ..])
            .axis_iter(Axis(1))
            .map(|x| x.to_vec())
            .collect();
        let pos_traily: Vec<Vec<f64>> = pos_trail_interim
            .slice(s![.., 1, ..])
            .axis_iter(Axis(1))
            .map(|x| x.to_vec())
            .collect();

        let ke_i = ke.slice(s![..=i]).to_vec();
        let pe_i = pe.slice(s![..=i]).to_vec();
        let t_all_i = t_all.slice(s![..=i]).to_vec();

        // Configure the curve
        let mut curve_points = Curve::new();
        let mut curve_trails = Curve::new();
        let mut curve_ke = Curve::new();
        let mut curve_pe = Curve::new();

        curve_points
            .set_line_style("None")
            .set_line_color("#35f0e0")
            .set_marker_color("#35f0e0")
            .set_marker_size(3.0)
            .set_marker_style("o");

        curve_trails
            .set_line_style("None")
            .set_line_color("#f035ea")
            .set_marker_color("#f035ea")
            .set_marker_size(1.0)
            .set_marker_style(".");

        // Draw curve
        curve_points.draw(&pos_x, &pos_y);

        (1..pos_trailx.len())
            .map(|j| {
                curve_trails.draw(&pos_trailx[j - 1], &pos_traily[j - 1]);
            })
            .count();

        // Draw energy curve
        curve_ke
            .set_label("KE")
            .set_line_alpha(0.8)
            .set_line_color("#2ca02c")
            .set_line_style("-")
            .draw(&t_all_i, &ke_i);

        curve_pe
            .set_label("PE")
            .set_line_alpha(0.8)
            .set_line_color("#d62728")
            .set_line_style("-")
            .draw(&t_all_i, &pe_i);

        // Add scatter plots
        // .extra("grid = plt.GridSpec(3, 1, wspace=0, hspace=0.7); plt.subplot(grid[0:2, 0])")
        //
        plot.set_subplot(2, 1, 1)
            .add(&curve_trails)
            .add(&curve_points)
            .set_labels("X", "Y")
            .set_range(-4.0, 4.0, -4.0, 4.0)
            .set_equal_axes(true);

        // add curve to subplot
        //
        // .extra("grid = plt.GridSpec(3, 1, wspace=0, hspace=0.7); plt.subplot(grid[2, 0])")
        plot.set_subplot(2, 1, 2)
            .add(&curve_ke)
            .add(&curve_pe)
            .set_range(0.0, tmax, e_range.0, e_range.1)
            .grid_labels_legend("Time", "Energy")
            .set_equal_axes(false);

        plot.save(&fname).unwrap_or(());
    });

    Ok(())
}

// ----------------------------------------------------------------
