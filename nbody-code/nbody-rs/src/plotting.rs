use color_eyre::eyre::{Error, Result};
use ndarray::{s, stack, Array1, Array3, Axis};
use ndarray_stats::QuantileExt;
use plotters::{prelude::*, style::RGBAColor};
// use ndarray::parallel::prelude::*;

macro_rules! energy_range {
    ($ke: expr, $pe: expr) => {
        (stack(Axis(0), &[$ke, $pe])?.min()?.to_owned())
            ..(stack(Axis(0), &[$ke, $pe])?.max()?.to_owned())
    };
}

pub fn plot_nbodysystem(
    pos: Array3<f64>,
    ke: Array1<f64>,
    pe: Array1<f64>,
    t_all: Array1<f64>,
    i: usize,
    filename: Option<&str>,
) -> Result<(), Error> {
    let filename = filename.unwrap_or(&"images/nbsys_{i}.png");

    let img_size: (u32, u32) = (320, 430);
    let root = BitMapBackend::new(filename, img_size).into_drawing_area();
    // let lfonts = ("Times New Roman", 10).into_font();
    let (upper, lower) = root.split_vertically(img_size.0);

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&upper)
        .margin(6)
        .set_all_label_area_size(30)
        .build_cartesian_2d(-4f64..4f64, -4f64..4f64)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("X axis")
        .y_desc("Y axis")
        .draw()?;

    chart.draw_series(
        pos.slice(s![.., .., i])
            .axis_iter(Axis(0))
            .map(|x| Circle::new((x[0], x[1]), 4, BLACK)),
    )?;

    // Lower part --------------------------------------------------------
    let x_axis = (0f64..t_all[t_all.len() - 1]).step(2.);
    let y_axis = energy_range!(ke.view(), pe.view()).step(5.);

    let mut chart = ChartBuilder::on(&lower)
        .margin(6)
        .set_all_label_area_size(30)
        .build_cartesian_2d(x_axis, y_axis)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Time")
        .y_desc("Energy")
        .draw()?;

    let ke_t = t_all.iter().zip(&ke);
    let pe_t = t_all.iter().zip(&pe);

    chart
        .draw_series(LineSeries::new(ke_t.map(|(x, y)| (*x, *y)), RED))?
        .label("KE")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(pe_t.map(|(x, y)| (*x, *y)), BLUE))?
        .label("PE")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // draw legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
