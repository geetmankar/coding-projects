// #![allow(dead_code, unused_imports, unused_variables)]
use color_eyre::eyre::{Error, Result};
use kdam::tqdm;
use ndarray::{s, Array1, Array3, Axis};
use ndarray_stats::QuantileExt;
use plotpy::{Curve, Plot};

macro_rules! energy_range {
    ($ke: expr, $pe: expr) => {
        (
            $pe.to_owned().min()?.to_owned(),
            $ke.to_owned().max()?.to_owned(),
        )
    };
}

fn max<T: PartialEq + PartialOrd>(x: T, y: T) -> T {
    match x > y {
        true => x,
        _ => y,
    }
}

// macro_rules! energy_range {
//     ($ke: expr, $pe: expr) => {
//         (
//             (stack(Axis(0), &[$ke, $pe])?.to_owned().min()?.to_owned()),
//             (stack(Axis(0), &[$ke, $pe])?.to_owned().max()?.to_owned()),
//         )
//     };
// }
// fn min<T: PartialEq + PartialOrd>(x: T, y: T) -> T {
//     if x < y {
//         x
//     } else {
//         y
//     }
// }
//ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>
//ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>
//ArrayBase<ViewRepr<f64>, Dim<[usize; 3]>>
//ArrayBase<ViewRepr<f64>, Dim<[usize; 1]>>

pub fn plot_nbodysystem(
    pos: Array3<f64>,
    ke: Array1<f64>,
    pe: Array1<f64>,
    t_all: Array1<f64>,
    filename: String,
) -> Result<(), Error> {
    let e_range = energy_range!(ke.view(), pe.view());

    let tmax = t_all[t_all.len() - 1];

    let mut plot = Plot::new();
    plot.set_figure_size_inches(4.0, 6.0).set_gaps(0.2, 0.2);

    for i in tqdm!(
        0..t_all.len(),
        desc = "Plotting...",
        colour = "green",
        unit = " frames"
    ) {
        if i > 0 {
            plot.clear_current_figure();
        };

        let fname = filename.clone() + &format!("_{i}.png");
        // let fname = fname.as_str();

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

        // .set_line_alpha(0.8)
        // .set_line_color("#1f77b4")
        curve_points
            .set_line_style("None")
            .set_marker_color("#1f77b4")
            .set_marker_size(3.0)
            .set_marker_style("o");

        curve_trails
            .set_line_style("None")
            .set_marker_color("#ff7f0e")
            .set_marker_size(1.0)
            .set_marker_style(".");

        // Draw curve
        curve_points.draw(&pos_x, &pos_y);

        for j in 1..pos_trailx.len() {
            curve_trails.draw(&pos_trailx[j - 1], &pos_traily[j - 1]);
        }
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
        plot.set_subplot(2, 1, 1)
            .add(&curve_points)
            .add(&curve_trails)
            .set_labels("X", "Y")
            .set_range(-4.0, 4.0, -4.0, 4.0)
            .set_equal_axes(true);

        // add curve to subplot
        plot.set_subplot(2, 1, 2)
            .add(&curve_ke)
            .add(&curve_pe)
            .set_range(0.0, tmax, e_range.0, e_range.1)
            .grid_labels_legend("Time", "Energy")
            .set_equal_axes(false);

        plot.save(&fname).unwrap_or(());
    }

    Ok(())
}

// ----------------------------------------------------------------
